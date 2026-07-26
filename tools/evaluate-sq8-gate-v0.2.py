#!/usr/bin/env python3
"""Consumer-side evaluator for the frozen SQ8_0 numerical gate v0.2.

The evaluator deliberately reads the frozen gate JSON on every invocation and
refuses a different byte sequence.  It consumes immutable F32 little-endian
captures, recomputes every metric in F64, and never uses producer-side pass/
fail fields.

Two input forms are supported:

* ``index-reference`` converts the resumable CPU reference directory to a
  read-only capture index outside that directory.
* ``evaluate`` compares that index with three control and two candidate
  ``capture-manifest.json`` files written by ``ullm-sq8-gate-capture``.

``--allow-incomplete-test-coverage`` exists only for harness self tests while
the reference is being generated.  Its receipt is deliberately labelled
``test_only_incomplete_coverage`` and can never produce an admission result.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import shutil
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


EXPECTED_GATE_SHA256 = "64a43c032570bed8086e3c441b0774cc470c5ab1e8c67f99e02af2b6307f72bf"
GATE_SCHEMA = "ullm.sq8.numerical_gate.relative_fp32.v0.2"
REFERENCE_INDEX_SCHEMA = "ullm.sq8.gate.v0.2.reference-index.v1"
CAPTURE_SCHEMA = "ullm.sq8.gate.v0.2.capture.v1"
RESULT_SCHEMA = "ullm.sq8.gate.v0.2.consumer-result.v1"
HIDDEN_SIZE = 5120
VOCAB_SIZE = 151936
LAYER_COUNT = 40


class GateError(RuntimeError):
    """An input-integrity or consumer-evaluation failure."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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
        raise GateError(f"cannot parse JSON {path}: {exc}") from exc


def write_json_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise GateError(f"refusing to overwrite existing output {path}")
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_frozen_gate(path: Path) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    digest = sha256_bytes(raw)
    if digest != EXPECTED_GATE_SHA256:
        raise GateError(
            "frozen gate SHA-256 mismatch: "
            f"expected={EXPECTED_GATE_SHA256} actual={digest}"
        )
    try:
        gate = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise GateError(f"frozen gate JSON cannot be parsed: {exc}") from exc
    if gate.get("schema_version") != GATE_SCHEMA:
        raise GateError(
            "frozen gate schema mismatch: "
            f"expected={GATE_SCHEMA} actual={gate.get('schema_version')!r}"
        )
    return gate, digest


def relative_path_or_absolute(path: Path) -> str:
    # Capture paths are intentionally absolute: a result receipt must remain
    # unambiguous when copied outside the worktree.
    return str(path.resolve())


def tensor_descriptor(path: Path, sha256: str, elements: int) -> dict[str, Any]:
    return {
        "path": relative_path_or_absolute(path),
        "sha256": sha256,
        "dtype": "f32le",
        "shape": [elements],
        "byte_count": elements * 4,
    }


def required_case_map(gate: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    corpus = gate["corpus"]
    values = list(corpus["primary_decode_streams"]) + list(corpus["required_boundary_cases"])
    result: dict[str, dict[str, Any]] = {}
    for case in values:
        case_id = case["id"]
        if case_id in result:
            raise GateError(f"duplicate frozen case id {case_id}")
        result[case_id] = dict(case)
    return result


def discover_reference_identity(case_root: Path) -> dict[str, Any]:
    plan = load_json(case_root / "plan.json")
    identity = plan.get("reference_identity")
    if not isinstance(identity, dict):
        raise GateError(f"reference case {case_root} has no reference_identity in plan.json")
    return {
        "artifact_content_sha256": identity.get("artifact_content_sha256"),
        "reference_executable_sha256": plan.get("executable_sha256"),
        "reference_schema_version": plan.get("reference_schema_version"),
        "fixture_manifest_sha256": plan.get("fixture_manifest_sha256"),
        "reference_identity": identity,
    }


def forward_directory(case_root: Path, ordinal: int, prompt_tokens: int) -> Path:
    if ordinal < prompt_tokens:
        return case_root / "forwards" / f"forward-{ordinal:05}-prompt-{ordinal:05}"
    decode_index = ordinal - prompt_tokens
    return case_root / "forwards" / f"forward-{ordinal:05}-decode-{decode_index:05}"


def load_reference_position(
    case_root: Path,
    case_id: str,
    mode: str,
    ordinal: int,
    phase: str,
    phase_index: int,
    tags: Sequence[str],
    layer_required: bool,
) -> dict[str, Any] | None:
    plan = load_json(case_root / "plan.json")
    prompt_tokens = int(plan["case"]["prompt_tokens"])
    capture_root = forward_directory(case_root, ordinal, prompt_tokens)
    metadata_path = capture_root / "metadata.json"
    if not metadata_path.is_file():
        return None
    metadata = load_json(metadata_path)
    forward = metadata.get("forward", {})
    files = metadata.get("files", {})
    if forward.get("position") != ordinal:
        raise GateError(
            f"reference capture ordinal mismatch at {metadata_path}: "
            f"expected={ordinal} actual={forward.get('position')!r}"
        )
    layer_hashes = forward.get("layer_hidden_f32le_sha256")
    if not isinstance(layer_hashes, list) or len(layer_hashes) != LAYER_COUNT:
        raise GateError(f"reference capture has invalid layer hash list: {metadata_path}")
    logits = capture_root / "logits.f32le"
    final_hidden = capture_root / "final-hidden.f32le"
    layers = []
    for layer_index, expected_hash in enumerate(layer_hashes):
        layer_path = capture_root / "layers" / f"layer-{layer_index:02}-hidden.f32le"
        recorded = files.get(f"layers/layer-{layer_index:02}-hidden.f32le")
        if recorded != expected_hash:
            raise GateError(
                f"reference layer hash disagreement at {metadata_path} layer={layer_index}"
            )
        layers.append(tensor_descriptor(layer_path, expected_hash, HIDDEN_SIZE))
    if files.get("logits.f32le") != forward.get("logits_f32le_sha256"):
        raise GateError(f"reference logits hash disagreement at {metadata_path}")
    if files.get("final-hidden.f32le") != forward.get("final_hidden_f32le_sha256"):
        raise GateError(f"reference final-hidden hash disagreement at {metadata_path}")
    return {
        "id": f"{mode}:{case_id}:{phase}:{phase_index:05}",
        "case_id": case_id,
        "mode": mode,
        "ordinal": ordinal,
        "phase": phase,
        "phase_index": phase_index,
        "position": int(forward["position"]),
        "input_token_id": int(forward["input_token_id"]),
        "reference_greedy_token_id": int(forward["greedy_token_id"]),
        "scope_tags": list(tags),
        "layer_required": layer_required,
        "logits": tensor_descriptor(logits, str(forward["logits_f32le_sha256"]), VOCAB_SIZE),
        "final_hidden": tensor_descriptor(
            final_hidden, str(forward["final_hidden_f32le_sha256"]), HIDDEN_SIZE
        ),
        "layers": layers,
        "capture_metadata_sha256": sha256_file(metadata_path),
    }


def hidden_probe_ids(gate: Mapping[str, Any]) -> set[tuple[str, int]]:
    """Return the frozen primary-decode layer-probe selection.

    The frozen JSON specifies the hash ordering but not an implementation
    language.  Concatenating the exact UTF-8 byte fragments below is the
    literal interpretation of that text; the receipt records it.
    """

    corpus = gate["corpus"]
    mandatory: set[tuple[str, int]] = set()
    primary = list(corpus["primary_decode_streams"])
    for stream in primary:
        mandatory.add((stream["id"], 0))
        mandatory.add((stream["id"], int(stream["forced_decode_tokens"]) - 1))

    candidates: list[tuple[bytes, str, int]] = []
    prefix = b"ullm.sq8.gate.v0.2.hidden-probe\0"
    for stream in primary:
        case_id = str(stream["id"])
        for decode_index in range(int(stream["forced_decode_tokens"])):
            if (case_id, decode_index) in mandatory:
                continue
            digest = hashlib.sha256(
                prefix + case_id.encode("utf-8") + b"\0" + f"{decode_index:05d}".encode("ascii")
            ).digest()
            candidates.append((digest, case_id, decode_index))
    candidates.sort()
    wanted = int(corpus["hidden_layer_probe"]["probe_count"])
    if len(mandatory) > wanted:
        raise GateError("frozen hidden mandatory set exceeds probe_count")
    selected = set(mandatory)
    for _, case_id, decode_index in candidates[: wanted - len(selected)]:
        selected.add((case_id, decode_index))
    if len(selected) != wanted:
        raise GateError("could not build frozen hidden probe selection")
    return selected


def expected_reference_case_keys(gate: Mapping[str, Any]) -> set[str]:
    corpus = gate["corpus"]
    sequential = [*corpus["primary_decode_streams"], *corpus["required_boundary_cases"]]
    values = {f"sequential_m1:{case['id']}" for case in sequential}
    values.update(
        f"m128_chunks_with_declared_tail:{case_id}"
        for case_id in corpus["prefill_coverage"]["m128_inputs"]
    )
    return values


def reference_case_completion(
    case_root: Path,
    mode: str,
    case: Mapping[str, Any],
    plan: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Read the immutable completion receipt without writing to the reference."""

    run_path = case_root / "run.json"
    if plan is None:
        return {"complete": False, "reason": "plan.json is unavailable"}
    if not run_path.is_file():
        return {"complete": False, "reason": "run.json is unavailable"}
    run = load_json(run_path)
    reasons: list[str] = []
    if run.get("status") != "complete":
        reasons.append(f"run status is {run.get('status')!r}")
    if run.get("mode") != mode:
        reasons.append(f"run mode is {run.get('mode')!r}")
    total_forwards = int(case["prompt_tokens"]) + int(case["forced_decode_tokens"])
    if run.get("total_forwards") != total_forwards:
        reasons.append(
            f"run total_forwards is {run.get('total_forwards')!r}, expected {total_forwards}"
        )
    plan_path = case_root / "plan.json"
    if run.get("plan_sha256") != sha256_file(plan_path):
        reasons.append("run plan_sha256 does not bind plan.json")
    teacher_path = case_root / "teacher-forced-tokens.u32le"
    if not teacher_path.is_file():
        reasons.append("teacher-forced-tokens.u32le is unavailable")
    elif run.get("teacher_forced_tokens_u32le_sha256") != sha256_file(teacher_path):
        reasons.append("run teacher-forced token SHA-256 does not match payload")
    return {
        "complete": not reasons,
        "run_sha256": sha256_file(run_path),
        "reason": "; ".join(reasons) if reasons else None,
    }


def index_reference(args: argparse.Namespace) -> int:
    gate, gate_sha = load_frozen_gate(args.gate)
    root = args.reference_root.resolve()
    if not root.is_dir():
        raise GateError(f"reference root is not a directory: {root}")
    cases = required_case_map(gate)
    probe_ids = hidden_probe_ids(gate)
    positions: list[dict[str, Any]] = []
    materialized_hashes: dict[str, str] = {}
    reference_identity: dict[str, Any] | None = None
    completion: dict[str, dict[str, Any]] = {}

    def add_case_positions(mode: str, case_id: str, purpose: str) -> None:
        nonlocal reference_identity
        case = cases[case_id]
        case_root = root / "cases" / mode / case_id
        plan_path = case_root / "plan.json"
        completion_key = f"{mode}:{case_id}"
        if not plan_path.is_file():
            completion[completion_key] = reference_case_completion(case_root, mode, case, None)
            return
        plan = load_json(plan_path)
        completion[completion_key] = reference_case_completion(case_root, mode, case, plan)
        materialized = plan.get("case", {}).get("materialized_input_sha256")
        if isinstance(materialized, str):
            materialized_hashes[f"{mode}:{case_id}"] = materialized
        identity = discover_reference_identity(case_root)
        if reference_identity is None:
            reference_identity = identity
        elif identity["artifact_content_sha256"] != reference_identity["artifact_content_sha256"]:
            raise GateError(f"reference artifact differs across cases at {case_root}")

        prompt_tokens = int(case["prompt_tokens"])
        forced = int(case["forced_decode_tokens"])
        if purpose in {"primary", "boundary"}:
            for decode_index in range(forced):
                tags = ["primary_decode", f"stream:{case_id}"] if purpose == "primary" else [
                    "boundary",
                    f"boundary:{case_id}",
                ]
                record = load_reference_position(
                    case_root,
                    case_id,
                    mode,
                    prompt_tokens + decode_index,
                    "decode",
                    decode_index,
                    tags,
                    (case_id, decode_index) in probe_ids or purpose == "boundary",
                )
                if record is not None:
                    positions.append(record)
        elif purpose == "prefill":
            checkpoint_path = case_root / "m128-checkpoints.json"
            if checkpoint_path.is_file():
                checkpoints = load_json(checkpoint_path).get("checkpoint_forward_indices", [])
            else:
                # A still-running reference has no checkpoint receipt yet.  Its
                # frozen plan nevertheless gives the same candidate-independent
                # set; only completed captures are indexed.
                checkpoints = plan.get("case", {}).get("m128_checkpoint_forward_indices", [])
            for ordinal in checkpoints:
                ordinal = int(ordinal)
                phase = "prompt" if ordinal < prompt_tokens else "decode"
                phase_index = ordinal if phase == "prompt" else ordinal - prompt_tokens
                record = load_reference_position(
                    case_root,
                    case_id,
                    mode,
                    ordinal,
                    phase,
                    phase_index,
                    ["prefill_checkpoint", f"prefill:{case_id}"],
                    True,
                )
                if record is not None:
                    positions.append(record)

    for stream in gate["corpus"]["primary_decode_streams"]:
        add_case_positions("sequential_m1", stream["id"], "primary")
    for boundary in gate["corpus"]["required_boundary_cases"]:
        add_case_positions("sequential_m1", boundary["id"], "boundary")
    for case_id in gate["corpus"]["prefill_coverage"]["m128_inputs"]:
        add_case_positions("m128_chunks_with_declared_tail", case_id, "prefill")

    if reference_identity is None:
        raise GateError(f"no readable reference plans under {root}")
    positions.sort(key=lambda item: item["id"])
    required_completion = expected_reference_case_keys(gate)
    completed_keys = {key for key, value in completion.items() if value.get("complete")}
    missing_completion = sorted(required_completion - completed_keys)
    index = {
        "schema_version": REFERENCE_INDEX_SCHEMA,
        "frozen_gate": {"path": relative_path_or_absolute(args.gate), "sha256": gate_sha},
        "reference_root": relative_path_or_absolute(root),
        "identity": {
            **reference_identity,
            "materialized_token_hashes": materialized_hashes,
            "device_identity": {"backend": "cpu_only_strict_f32"},
            "runtime_compiler_versions": {"status": "recorded_by_reference_plan"},
        },
        "probe_selection": {
            "count": len(probe_ids),
            "hash_input": "utf8(seed_domain_with_trailing_nul + case_id + nul + zero_padded_decode_index)",
            "quantile_or_rng_assumptions": [],
        },
        "reference_qualification": {
            "complete": not missing_completion,
            "required_case_count": len(required_completion),
            "completed_case_count": len(completed_keys),
            "missing_cases": missing_completion,
            "case_completion": completion,
        },
        "positions": positions,
        "indexed_at": "consumer_side_read_only",
    }
    write_json_new(args.output, index)
    print(json.dumps({"output": str(args.output), "positions": len(positions)}, sort_keys=True))
    return 0


def ensure_capture_manifest(value: Any, path: Path, role: str, gate_sha: str) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema_version") != CAPTURE_SCHEMA:
        raise GateError(f"{role} capture has unsupported schema at {path}")
    frozen = value.get("frozen_gate")
    if not isinstance(frozen, dict) or frozen.get("sha256") != gate_sha:
        raise GateError(f"{role} capture gate hash does not match frozen gate: {path}")
    if value.get("role") != role:
        raise GateError(f"capture role mismatch at {path}: expected={role} actual={value.get('role')!r}")
    if not isinstance(value.get("positions"), list):
        raise GateError(f"capture positions are missing at {path}")
    return value


def index_positions(value: Mapping[str, Any], label: str) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    for record in value["positions"]:
        if not isinstance(record, dict) or not isinstance(record.get("id"), str):
            raise GateError(f"{label} contains a position without an id")
        position_id = record["id"]
        if position_id in values:
            raise GateError(f"{label} repeats capture position {position_id}")
        values[position_id] = record
    return values


def check_identity(reference: Mapping[str, Any], runs: Sequence[Mapping[str, Any]]) -> list[str]:
    errors: list[str] = []
    reference_identity = reference.get("identity", {})
    keys = (
        "artifact_content_sha256",
        "fixture_manifest_sha256",
        "materialized_token_hashes",
        "reference_executable_sha256",
    )
    for run in runs:
        identity = run.get("identity")
        if not isinstance(identity, dict):
            errors.append("capture has no identity object")
            continue
        for key in keys:
            expected = reference_identity.get(key)
            actual = identity.get(key)
            if expected is None or actual is None:
                errors.append(f"identity field missing: {key}")
            elif expected != actual:
                errors.append(f"identity mismatch {key}: expected={expected!r} actual={actual!r}")
        for key in (
            "executable_sha256",
            "selector_configuration_fingerprint",
            "device_identity",
            "runtime_compiler_versions",
        ):
            if key not in identity:
                errors.append(f"capture identity does not record required field {key}")
    return errors


def check_matched_capture_configuration(runs: Sequence[tuple[str, Mapping[str, Any]]]) -> list[str]:
    """Require a control/candidate match on every non-selector GPU setting."""

    if not runs:
        return ["no capture manifests supplied for runtime configuration matching"]
    baseline_label, baseline = runs[0]
    baseline_identity = baseline.get("identity")
    if not isinstance(baseline_identity, dict):
        return [f"{baseline_label} has no identity object for runtime configuration matching"]
    keys = (
        "executable_sha256",
        "device_identity",
        "runtime_compiler_versions",
        "hip_guard_environment",
    )
    errors: list[str] = []
    for key in keys:
        expected = baseline_identity.get(key)
        if expected is None:
            errors.append(f"{baseline_label} does not record matched runtime field {key}")
            continue
        for label, run in runs[1:]:
            identity = run.get("identity")
            actual = identity.get(key) if isinstance(identity, dict) else None
            if actual is None:
                errors.append(f"{label} does not record matched runtime field {key}")
            elif actual != expected:
                errors.append(
                    f"matched runtime mismatch {key}: baseline={baseline_label} "
                    f"expected={expected!r} {label} actual={actual!r}"
                )
    return errors


@dataclasses.dataclass
class TensorRead:
    values: np.ndarray
    descriptor: Mapping[str, Any]


class TensorReader:
    def __init__(self, verify_hashes: bool) -> None:
        self.verify_hashes = verify_hashes
        self._hash_cache: dict[str, str] = {}

    def read(self, descriptor: Mapping[str, Any], expected_elements: int, label: str) -> TensorRead:
        if descriptor.get("dtype") != "f32le":
            raise GateError(f"{label} dtype must be f32le, got {descriptor.get('dtype')!r}")
        if descriptor.get("shape") != [expected_elements]:
            raise GateError(f"{label} shape must be [{expected_elements}], got {descriptor.get('shape')!r}")
        path = Path(str(descriptor.get("path", "")))
        if not path.is_file():
            raise GateError(f"{label} tensor is missing: {path}")
        expected_bytes = expected_elements * 4
        actual_bytes = path.stat().st_size
        if descriptor.get("byte_count") != expected_bytes or actual_bytes != expected_bytes:
            raise GateError(
                f"{label} byte count mismatch: descriptor={descriptor.get('byte_count')!r} "
                f"actual={actual_bytes} expected={expected_bytes}"
            )
        expected_hash = descriptor.get("sha256")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise GateError(f"{label} has no SHA-256")
        if self.verify_hashes:
            key = str(path.resolve())
            actual_hash = self._hash_cache.get(key)
            if actual_hash is None:
                actual_hash = sha256_file(path)
                self._hash_cache[key] = actual_hash
            if actual_hash != expected_hash:
                raise GateError(
                    f"{label} SHA-256 mismatch: expected={expected_hash} actual={actual_hash} path={path}"
                )
        try:
            values = np.fromfile(path, dtype="<f4")
        except OSError as exc:
            raise GateError(f"cannot read {label}: {exc}") from exc
        if values.size != expected_elements:
            raise GateError(f"{label} element count changed while reading")
        if not np.isfinite(values).all():
            index = int(np.flatnonzero(~np.isfinite(values))[0])
            raise GateError(f"{label} has non-finite F32 value at element {index}")
        return TensorRead(values=values, descriptor=descriptor)


@dataclasses.dataclass
class VectorMetrics:
    relative_l2: float
    max_abs: float
    kl: float | None
    top1: int | None
    top10_contains_reference_top1: bool | None
    reference_top1: int | None
    reference_top2: int | None
    reference_margin: float | None
    reference_scale: float
    squared_error: float
    reference_squared_l2: float


def top_ids(values: np.ndarray, count: int) -> np.ndarray:
    if count <= 0 or count > values.size:
        raise GateError("invalid top-k count")
    # ``argpartition`` avoids sorting a 151,936-token vocabulary at every
    # position.  The final lexsort makes the frozen tie-break explicit.
    selected = np.argpartition(-values, count - 1)[:count]
    boundary = values[selected].min()
    tied = np.flatnonzero(values >= boundary)
    ordered = tied[np.lexsort((tied, -values[tied]))]
    return ordered[:count]


def kl_reference_to_actual(reference: np.ndarray, actual: np.ndarray) -> float:
    ref64 = reference.astype(np.float64, copy=False)
    actual64 = actual.astype(np.float64, copy=False)
    ref_max = float(np.max(ref64))
    actual_max = float(np.max(actual64))
    ref_exp = np.exp(ref64 - ref_max)
    actual_exp = np.exp(actual64 - actual_max)
    ref_logsumexp = ref_max + math.log(float(np.sum(ref_exp, dtype=np.float64)))
    actual_logsumexp = actual_max + math.log(float(np.sum(actual_exp, dtype=np.float64)))
    probabilities = ref_exp / float(np.sum(ref_exp, dtype=np.float64))
    value = float(
        np.sum(probabilities * ((ref64 - ref_logsumexp) - (actual64 - actual_logsumexp)), dtype=np.float64)
    )
    # Floating point cancellation can produce a tiny negative KL.  It is an
    # F64 calculation error, not a negative divergence.
    return max(0.0, value)


def vector_metrics(reference: np.ndarray, actual: np.ndarray, include_logits: bool) -> VectorMetrics:
    difference = actual.astype(np.float64, copy=False) - reference.astype(np.float64, copy=False)
    numerator = float(np.sum(difference * difference, dtype=np.float64))
    reference_squared_l2 = float(np.sum(reference.astype(np.float64, copy=False) ** 2, dtype=np.float64))
    relative_l2 = math.sqrt(numerator / max(reference_squared_l2, 1.0e-30))
    max_abs = float(np.max(np.abs(difference)))
    reference_scale = float(np.max(np.abs(reference.astype(np.float64, copy=False))))
    if not include_logits:
        return VectorMetrics(
            relative_l2,
            max_abs,
            None,
            None,
            None,
            None,
            None,
            None,
            reference_scale,
            numerator,
            reference_squared_l2,
        )
    reference_top = top_ids(reference, 10)
    actual_top = top_ids(actual, 10)
    top1 = int(actual_top[0])
    reference_top1 = int(reference_top[0])
    return VectorMetrics(
        relative_l2=relative_l2,
        max_abs=max_abs,
        kl=kl_reference_to_actual(reference, actual),
        top1=top1,
        top10_contains_reference_top1=reference_top1 in set(int(value) for value in actual_top),
        reference_top1=reference_top1,
        reference_top2=int(reference_top[1]),
        reference_margin=float(reference[reference_top[0]] - reference[reference_top[1]]),
        reference_scale=reference_scale,
        squared_error=numerator,
        reference_squared_l2=reference_squared_l2,
    )


def nearest_rank(values: Sequence[float], probability: float) -> tuple[float, int]:
    if not values:
        raise GateError("cannot calculate a quantile of zero values")
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    rank = max(1, math.ceil(probability * len(ordered)))
    original_index, value = ordered[rank - 1]
    return float(value), original_index


def ulp_f32(value: float) -> float:
    base = np.float32(max(1.0, value))
    return float(np.nextafter(base, np.float32(np.inf), dtype=np.float32) - base)


def wilson_lower(successes: int, total: int, confidence: float) -> float:
    if total <= 0:
        raise GateError("Wilson interval requires a positive sample count")
    if successes < 0 or successes > total:
        raise GateError("Wilson successes outside sample count")
    z = NormalDist().inv_cdf(confidence)
    p = successes / total
    denominator = 1.0 + z * z / total
    centre = p + z * z / (2.0 * total)
    radius = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total)
    return (centre - radius) / denominator


def as_upper_gate(
    values_control: Sequence[float],
    values_candidate: Sequence[float],
    factor: float,
    floor: float,
) -> dict[str, Any]:
    if len(values_control) != 3 or len(values_candidate) != 2:
        raise GateError("upper gate requires exactly three control and two candidate repetitions")
    median = float(statistics.median(values_control))
    envelope = max(values_control) - median
    worst = max(values_candidate)
    threshold = median * factor + envelope + floor
    return {
        "control_values": list(values_control),
        "candidate_values": list(values_candidate),
        "control_median": median,
        "repeat_envelope": envelope,
        "candidate_worst": worst,
        "absolute_floor": floor,
        "threshold": threshold,
        "passed": worst <= threshold,
    }


def as_lower_gate(
    values_control: Sequence[float],
    values_candidate: Sequence[float],
    margin: float,
) -> dict[str, Any]:
    if len(values_control) != 3 or len(values_candidate) != 2:
        raise GateError("lower gate requires exactly three control and two candidate repetitions")
    median = float(statistics.median(values_control))
    envelope = median - min(values_control)
    worst = min(values_candidate)
    threshold = median - envelope - margin
    return {
        "control_values": list(values_control),
        "candidate_values": list(values_candidate),
        "control_median": median,
        "repeat_envelope": envelope,
        "candidate_worst": worst,
        "noninferiority_margin": margin,
        "threshold": threshold,
        "passed": worst >= threshold,
    }


def scope_definitions(gate: Mapping[str, Any], positions: Mapping[str, Mapping[str, Any]]) -> dict[str, list[str]]:
    values: dict[str, list[str]] = {}
    primary = [position_id for position_id, record in positions.items() if "primary_decode" in record["scope_tags"]]
    values["aggregate_primary_decode"] = sorted(primary)
    for stream in gate["corpus"]["primary_decode_streams"]:
        name = f"stream:{stream['id']}"
        values[f"each_primary_decode_stream:{stream['id']}"] = sorted(
            position_id for position_id, record in positions.items() if name in record["scope_tags"]
        )
    for boundary in gate["corpus"]["required_boundary_cases"]:
        name = f"boundary:{boundary['id']}"
        values[f"each_required_boundary_case:{boundary['id']}"] = sorted(
            position_id for position_id, record in positions.items() if name in record["scope_tags"]
        )
    values["prefill_checkpoint_set"] = sorted(
        position_id for position_id, record in positions.items() if "prefill_checkpoint" in record["scope_tags"]
    )
    return values


def position_id(mode: str, case_id: str, phase: str, phase_index: int) -> str:
    """Return the stable ID used by both the reference index and capture manifests."""

    return f"{mode}:{case_id}:{phase}:{phase_index:05}"


def expected_prefill_checkpoint_ids(gate: Mapping[str, Any]) -> set[str]:
    """Derive every frozen M=128 checkpoint without consulting a producer plan.

    The frozen JSON says that checkpoints occur after every complete M=128
    chunk and after a final tail.  Deriving this set from the frozen corpus is
    deliberately stricter than accepting whichever checkpoints a producer
    happened to record in its mutable plan receipt.
    """

    cases = required_case_map(gate)
    expected: set[str] = set()
    for case_id in gate["corpus"]["prefill_coverage"]["m128_inputs"]:
        case = cases[case_id]
        prompt_tokens = int(case["prompt_tokens"])
        total_tokens = prompt_tokens + int(case["forced_decode_tokens"])
        ordinals = list(range(127, total_tokens, 128))
        if not ordinals or ordinals[-1] != total_tokens - 1:
            ordinals.append(total_tokens - 1)
        for ordinal in ordinals:
            if ordinal < prompt_tokens:
                expected.add(position_id("m128_chunks_with_declared_tail", case_id, "prompt", ordinal))
            else:
                expected.add(
                    position_id(
                        "m128_chunks_with_declared_tail",
                        case_id,
                        "decode",
                        ordinal - prompt_tokens,
                    )
                )
    return expected


def expected_position_sets(gate: Mapping[str, Any]) -> dict[str, set[str]]:
    """Return the exact frozen position sets used for qualification coverage."""

    primary: set[str] = set()
    boundaries: set[str] = set()
    for stream in gate["corpus"]["primary_decode_streams"]:
        primary.update(
            position_id("sequential_m1", stream["id"], "decode", decode_index)
            for decode_index in range(int(stream["forced_decode_tokens"]))
        )
    for boundary in gate["corpus"]["required_boundary_cases"]:
        boundaries.update(
            position_id("sequential_m1", boundary["id"], "decode", decode_index)
            for decode_index in range(int(boundary["forced_decode_tokens"]))
        )

    primary_layer = {
        position_id("sequential_m1", case_id, "decode", decode_index)
        for case_id, decode_index in hidden_probe_ids(gate)
    }
    prefill = expected_prefill_checkpoint_ids(gate)
    return {
        "primary": primary,
        "boundaries": boundaries,
        "prefill": prefill,
        "layer_required": primary_layer | boundaries | prefill,
    }


def coverage_delta(actual: set[str], expected: set[str]) -> dict[str, Any]:
    """Compact audit information without putting thousands of IDs in receipts."""

    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    return {
        "actual": len(actual),
        "required": len(expected),
        "missing": len(missing),
        "unexpected": len(unexpected),
        "missing_examples": missing[:5],
        "unexpected_examples": unexpected[:5],
    }


def index_reference_qualification(gate: Mapping[str, Any], index: Mapping[str, Any]) -> dict[str, Any]:
    """Validate that the index binds every completed strict-F32 reference case."""

    value = index.get("reference_qualification")
    required = expected_reference_case_keys(gate)
    if not isinstance(value, dict):
        return {
            "complete": False,
            "reason": "reference index does not record reference_qualification",
            "required_case_count": len(required),
        }
    case_completion = value.get("case_completion")
    if not isinstance(case_completion, dict):
        return {
            "complete": False,
            "reason": "reference index does not record per-case completion",
            "required_case_count": len(required),
        }
    completed = {
        key
        for key, receipt in case_completion.items()
        if isinstance(receipt, dict) and receipt.get("complete") is True
    }
    missing = sorted(required - completed)
    return {
        "complete": not missing and value.get("complete") is True,
        "required_case_count": len(required),
        "completed_case_count": len(completed),
        "missing_cases": missing,
        "reason": None if not missing and value.get("complete") is True else "strict-F32 reference completion is incomplete",
    }


def expected_coverage(gate: Mapping[str, Any], positions: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    corpus = gate["corpus"]
    expected = expected_position_sets(gate)
    primary = {
        position_id
        for position_id, record in positions.items()
        if "primary_decode" in record["scope_tags"]
    }
    streams = {positions[position_id]["case_id"] for position_id in primary}
    boundaries = {
        position_id
        for position_id, record in positions.items()
        if "boundary" in record["scope_tags"]
    }
    prefill = {
        position_id
        for position_id, record in positions.items()
        if "prefill_checkpoint" in record["scope_tags"]
    }
    layer_ids = {position_id for position_id, record in positions.items() if record.get("layer_required")}
    primary_delta = coverage_delta(primary, expected["primary"])
    boundary_delta = coverage_delta(boundaries, expected["boundaries"])
    prefill_delta = coverage_delta(prefill, expected["prefill"])
    layer_delta = coverage_delta(layer_ids, expected["layer_required"])
    return {
        "actual": {
            "primary_decode_positions": len(primary),
            "primary_decode_streams": len(streams),
            "decode_blocks_of_64": len(primary) // 64,
            "hidden_layer_probe_or_mandatory_positions": len(layer_ids),
            "mandatory_boundary_positions": len(boundaries),
            "prefill_checkpoints": len(prefill),
        },
        "required": {
            **dict(gate["qualification_and_decision"]["minimum_coverage"]),
            "mandatory_boundary_positions": len(expected["boundaries"]),
            "prefill_checkpoints": len(expected["prefill"]),
            "hidden_layer_probe_or_mandatory_positions": len(expected["layer_required"]),
        },
        "position_sets": {
            "primary_decode": primary_delta,
            "required_boundaries": boundary_delta,
            "m128_checkpoints": prefill_delta,
            "layer_hidden": layer_delta,
        },
        "complete": (
            primary == expected["primary"]
            and len(streams) == len(corpus["primary_decode_streams"])
            and len(primary) // 64 == int(gate["qualification_and_decision"]["minimum_coverage"]["decode_blocks_of_64"])
            and boundaries == expected["boundaries"]
            and prefill == expected["prefill"]
            and layer_ids == expected["layer_required"]
        ),
    }


def collect_run_metrics(
    reader: TensorReader,
    reference: Mapping[str, Mapping[str, Any]],
    run: Mapping[str, Mapping[str, Any]],
    position_ids: Iterable[str],
    require_layers: bool,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for position_id in position_ids:
        ref = reference[position_id]
        actual = run.get(position_id)
        if actual is None:
            raise GateError(f"candidate/control is missing reference position {position_id}")
        for key in ("case_id", "mode", "ordinal", "phase", "phase_index", "input_token_id"):
            if actual.get(key) != ref.get(key):
                raise GateError(
                    f"position identity mismatch at {position_id} field={key}: "
                    f"reference={ref.get(key)!r} actual={actual.get(key)!r}"
                )
        logits = vector_metrics(
            reader.read(ref["logits"], VOCAB_SIZE, f"reference logits {position_id}").values,
            reader.read(actual["logits"], VOCAB_SIZE, f"run logits {position_id}").values,
            True,
        )
        final_hidden = vector_metrics(
            reader.read(ref["final_hidden"], HIDDEN_SIZE, f"reference final hidden {position_id}").values,
            reader.read(actual["final_hidden"], HIDDEN_SIZE, f"run final hidden {position_id}").values,
            False,
        )
        layers: list[VectorMetrics] = []
        if require_layers and ref.get("layer_required"):
            actual_layers = actual.get("layers")
            if not isinstance(actual_layers, list) or len(actual_layers) != LAYER_COUNT:
                raise GateError(f"run is missing all layer captures at {position_id}")
            for layer_index in range(LAYER_COUNT):
                layers.append(
                    vector_metrics(
                        reader.read(ref["layers"][layer_index], HIDDEN_SIZE, f"reference layer {layer_index} {position_id}").values,
                        reader.read(actual_layers[layer_index], HIDDEN_SIZE, f"run layer {layer_index} {position_id}").values,
                        False,
                    )
                )
        output[position_id] = {"logits": logits, "final_hidden": final_hidden, "layers": layers}
    return output


def aggregate_continuous(
    per_position: Mapping[str, Mapping[str, Any]],
    position_ids: Sequence[str],
    key: str,
    include_kl: bool,
) -> dict[str, Any]:
    values = [per_position[position_id][key] for position_id in position_ids]
    if not values:
        raise GateError(f"scope has no positions for {key}")
    # Aggregate relative L2 is not the mean of position L2.  Reconstructing
    # raw numerator/denominator would retain every vector; derive it here only
    # when the capture values are all deterministic F32.  The evaluator keeps
    # exact aggregate values separately below using tensor-level accumulation.
    rel_values = [metric.relative_l2 for metric in values]
    max_values = [metric.max_abs for metric in values]
    scales = [metric.reference_scale for metric in values]
    p99_rel, p99_index = nearest_rank(rel_values, 0.99)
    max_index = max(range(len(max_values)), key=max_values.__getitem__)
    result = {
        "p99_position_relative_l2": p99_rel,
        "p99_position_relative_l2_position": position_ids[p99_index],
        "max_abs": max(max_values),
        "max_abs_position": position_ids[max_index],
        "reference_scale": max(scales),
        # This field is replaced by a proper aggregate from the retained F64
        # numerator/denominator below.
        "aggregate_relative_l2": None,
    }
    if include_kl:
        kl_values = [float(metric.kl) for metric in values]
        p99_kl, p99_kl_index = nearest_rank(kl_values, 0.99)
        result.update(
            {
                "mean_kl_nats": float(statistics.fmean(kl_values)),
                "p99_kl_nats": p99_kl,
                "p99_kl_nats_position": position_ids[p99_kl_index],
            }
        )
    return result


def aggregate_metric_relative_l2(
    per_position: Mapping[str, Mapping[str, Any]],
    position_ids: Sequence[str],
    tensor_key: str,
) -> float:
    """Aggregate already-recomputed F64 vector sums without rereading tensors."""

    squared_error = 0.0
    reference_squared_l2 = 0.0
    for position_id in position_ids:
        metric = per_position[position_id][tensor_key]
        squared_error += metric.squared_error
        reference_squared_l2 += metric.reference_squared_l2
    return math.sqrt(squared_error / max(reference_squared_l2, 1.0e-30))


def policy_agreement(
    ref_metric: VectorMetrics,
    candidate_metric: VectorMetrics,
    control_metrics: Sequence[VectorMetrics],
    allow_near_margin: bool,
) -> tuple[bool, bool]:
    if candidate_metric.top1 == ref_metric.reference_top1:
        return True, False
    candidate_is_ref_top2 = candidate_metric.top1 == ref_metric.reference_top2
    control_max_abs = max(metric.max_abs for metric in control_metrics)
    envelope = 2.0 * (control_max_abs + 16.0 * ulp_f32(ref_metric.reference_scale))
    near = candidate_is_ref_top2 and float(ref_metric.reference_margin) <= envelope and allow_near_margin
    return near, near


def evaluate_discrete_scope(
    gate: Mapping[str, Any],
    scope_ids: Sequence[str],
    controls: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    continuous_and_top10_pass: bool,
) -> dict[str, Any]:
    confidence = float(gate["noninferiority_rule"]["discrete"]["position_wilson"]["confidence"])
    margin = float(gate["noninferiority_rule"]["discrete"]["position_wilson"]["noninferiority_margin"])
    control_wilson_top1: list[float] = []
    control_wilson_top10: list[float] = []
    candidate_wilson_top1: list[float] = []
    candidate_wilson_top10: list[float] = []
    hard_regressions: list[dict[str, Any]] = []
    candidate_agreements_by_run: list[list[int]] = []
    control_agreements_by_run: list[list[int]] = []
    candidate_policy_mismatch_positions: list[list[str]] = []
    candidate_top10_mismatch_positions: list[list[str]] = []

    for control in controls:
        agreement = [
            int(control[position_id]["logits"].top1 == controls[0][position_id]["logits"].reference_top1)
            for position_id in scope_ids
        ]
        top10 = [
            int(bool(control[position_id]["logits"].top10_contains_reference_top1)) for position_id in scope_ids
        ]
        control_agreements_by_run.append(agreement)
        control_wilson_top1.append(wilson_lower(sum(agreement), len(agreement), confidence))
        control_wilson_top10.append(wilson_lower(sum(top10), len(top10), confidence))

    for candidate_index, candidate in enumerate(candidates):
        agreement: list[int] = []
        top10: list[int] = []
        policy_mismatches: list[str] = []
        top10_mismatches: list[str] = []
        for position_id in scope_ids:
            ref_metric = controls[0][position_id]["logits"]
            candidate_metric = candidate[position_id]["logits"]
            control_metrics = [control[position_id]["logits"] for control in controls]
            allowed, was_near = policy_agreement(
                ref_metric, candidate_metric, control_metrics, continuous_and_top10_pass
            )
            agreement.append(int(allowed))
            top10.append(int(bool(candidate_metric.top10_contains_reference_top1)))
            if not allowed:
                policy_mismatches.append(position_id)
            if not candidate_metric.top10_contains_reference_top1:
                top10_mismatches.append(position_id)
            all_controls_agree = all(metric.top1 == ref_metric.reference_top1 for metric in control_metrics)
            if candidate_metric.top1 != ref_metric.reference_top1 and all_controls_agree and not was_near:
                hard_regressions.append(
                    {
                        "candidate_repetition": candidate_index,
                        "position": position_id,
                        "reference_top1": ref_metric.reference_top1,
                        "candidate_top1": candidate_metric.top1,
                        "reference_top2": ref_metric.reference_top2,
                        "reference_margin": ref_metric.reference_margin,
                    }
                )
        candidate_agreements_by_run.append(agreement)
        candidate_policy_mismatch_positions.append(policy_mismatches)
        candidate_top10_mismatch_positions.append(top10_mismatches)
        candidate_wilson_top1.append(wilson_lower(sum(agreement), len(agreement), confidence))
        candidate_wilson_top10.append(wilson_lower(sum(top10), len(top10), confidence))

    return {
        "top1_wilson": as_lower_gate(control_wilson_top1, candidate_wilson_top1, margin),
        "top10_wilson": as_lower_gate(control_wilson_top10, candidate_wilson_top10, margin),
        "hard_top1_regressions": hard_regressions,
        "control_policy_agreement": control_agreements_by_run,
        "candidate_policy_agreement": candidate_agreements_by_run,
        "candidate_policy_mismatch_positions": candidate_policy_mismatch_positions,
        "candidate_top10_mismatch_positions": candidate_top10_mismatch_positions,
    }


def bootstrap_lower_bound(
    gate: Mapping[str, Any],
    primary_ids: Sequence[str],
    positions: Mapping[str, Mapping[str, Any]],
    control_agreement: Sequence[int],
    candidate_agreement: Sequence[int],
) -> dict[str, Any]:
    config = gate["noninferiority_rule"]["discrete"]["block_bootstrap"]
    block_size = int(config["block_size_decode_positions"])
    replicates = int(config["replicates"])
    confidence = float(config["confidence"])
    grouped: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for position_id, control, candidate in zip(primary_ids, control_agreement, candidate_agreement, strict=True):
        grouped[positions[position_id]["case_id"]].append((control, candidate))
    blocks: list[np.ndarray] = []
    for case_id in sorted(grouped):
        values = grouped[case_id]
        if len(values) % block_size != 0:
            raise GateError(f"primary stream {case_id} has partial {block_size}-token bootstrap block")
        array = np.asarray(values, dtype=np.float64)
        blocks.append(array.reshape((-1, block_size, 2)).sum(axis=1))
    if sum(block.shape[0] for block in blocks) != int(config["primary_block_count"]):
        raise GateError("bootstrap block count does not match frozen JSON")
    seed_bytes = hashlib.sha256(str(config["seed_domain"]).encode("utf-8")).digest()
    seed = int.from_bytes(seed_bytes[:8], "big", signed=False)
    rng = np.random.Generator(np.random.PCG64(seed))
    results = np.empty(replicates, dtype=np.float64)
    total = len(primary_ids)
    for replicate in range(replicates):
        control_sum = 0.0
        candidate_sum = 0.0
        for stream_blocks in blocks:
            selected = rng.integers(0, stream_blocks.shape[0], size=stream_blocks.shape[0])
            totals = stream_blocks[selected].sum(axis=0)
            control_sum += float(totals[0])
            candidate_sum += float(totals[1])
        results[replicate] = (candidate_sum - control_sum) / total
    lower, _ = nearest_rank(results.tolist(), 1.0 - confidence)
    threshold = float(config["require_lower_bound_at_least"])
    return {
        "applicable": True,
        "lower_bound": lower,
        "threshold": threshold,
        "passed": lower >= threshold,
        "replicates": replicates,
        "seed": seed,
        "rng": "numpy.PCG64",
        "control_minus_candidate_definition": "candidate policy-aware top1 agreement rate minus median-control-run policy-aware agreement rate",
    }


def evaluate(args: argparse.Namespace) -> int:
    gate, gate_sha = load_frozen_gate(args.gate)
    reference_value = load_json(args.reference)
    if reference_value.get("schema_version") != REFERENCE_INDEX_SCHEMA:
        raise GateError(f"reference index has unsupported schema: {args.reference}")
    if reference_value.get("frozen_gate", {}).get("sha256") != gate_sha:
        raise GateError("reference index does not bind this frozen gate")
    controls_values = [ensure_capture_manifest(load_json(path), path, "control", gate_sha) for path in args.control]
    candidates_values = [ensure_capture_manifest(load_json(path), path, "candidate", gate_sha) for path in args.candidate]
    if len(controls_values) != int(gate["capture_contract"]["control"]["repetitions"]):
        raise GateError("frozen gate requires exactly three controls")
    if len(candidates_values) != int(gate["capture_contract"]["candidate"]["repetitions"]):
        raise GateError("frozen gate requires exactly two candidates")
    identity_errors = check_identity(reference_value, [*controls_values, *candidates_values])
    identity_errors.extend(
        check_matched_capture_configuration(
            [
                *[(f"control-{index}", value) for index, value in enumerate(controls_values)],
                *[(f"candidate-{index}", value) for index, value in enumerate(candidates_values)],
            ]
        )
    )
    for control in controls_values:
        selector = control.get("selector")
        if not isinstance(selector, dict) or selector.get("enabled") is not False:
            identity_errors.append("control selector is not explicitly disabled")
    candidate_ids = {value.get("candidate", {}).get("id") for value in candidates_values}
    if len(candidate_ids) != 1 or None in candidate_ids:
        identity_errors.append("candidate repetitions do not share one candidate id")
    for candidate in candidates_values:
        selector = candidate.get("selector")
        if not isinstance(selector, dict) or selector.get("enabled") is not True:
            identity_errors.append("candidate selector is not explicitly enabled")

    reference_positions = index_positions(reference_value, "reference")
    controls_positions = [index_positions(value, f"control-{index}") for index, value in enumerate(controls_values)]
    candidates_positions = [index_positions(value, f"candidate-{index}") for index, value in enumerate(candidates_values)]
    required_ids = sorted(reference_positions)
    missing = {
        f"control-{index}": sorted(set(required_ids) - set(value)) for index, value in enumerate(controls_positions)
    }
    missing.update(
        {f"candidate-{index}": sorted(set(required_ids) - set(value)) for index, value in enumerate(candidates_positions)}
    )
    missing = {name: values for name, values in missing.items() if values}
    if missing:
        identity_errors.append(f"capture positions missing: {missing}")
    unexpected = {
        f"control-{index}": sorted(set(value) - set(required_ids)) for index, value in enumerate(controls_positions)
    }
    unexpected.update(
        {f"candidate-{index}": sorted(set(value) - set(required_ids)) for index, value in enumerate(candidates_positions)}
    )
    unexpected = {name: values for name, values in unexpected.items() if values}
    if unexpected:
        identity_errors.append(f"capture contains positions outside the reference index: {unexpected}")
    test_only_manifests = [
        f"control-{index}" for index, value in enumerate(controls_values) if value.get("test_only")
    ] + [
        f"candidate-{index}" for index, value in enumerate(candidates_values) if value.get("test_only")
    ]
    if test_only_manifests and not args.allow_incomplete_test_coverage:
        identity_errors.append(
            "test-only capture manifests require --allow-incomplete-test-coverage: "
            + ", ".join(test_only_manifests)
        )

    coverage = expected_coverage(gate, reference_positions)
    reference_qualification = index_reference_qualification(gate, reference_value)
    coverage["reference_qualification"] = reference_qualification
    coverage["complete"] = bool(coverage["complete"] and reference_qualification["complete"])
    if identity_errors:
        result = {
            "schema_version": RESULT_SCHEMA,
            "status": "fail_relative_fp32_v0_2",
            "frozen_gate_sha256": gate_sha,
            "identity_errors": identity_errors,
            "coverage": coverage,
        }
        write_json_new(args.output_json, result)
        write_markdown(args.output_markdown, result)
        return 1
    if not coverage["complete"] and not args.allow_incomplete_test_coverage:
        result = {
            "schema_version": RESULT_SCHEMA,
            "status": "blocked_reference_or_capture",
            "frozen_gate_sha256": gate_sha,
            "identity_errors": [],
            "coverage": coverage,
            "reason": "frozen coverage or strict-F32 reference qualification is incomplete; use the test-only switch only for harness verification",
        }
        write_json_new(args.output_json, result)
        write_markdown(args.output_markdown, result)
        return 2

    reader = TensorReader(verify_hashes=not args.skip_payload_hash_verification)
    # All positions are computed for logits/final hidden; layer metrics use the
    # frozen probe plus mandatory positions only.
    controls_metrics = [
        collect_run_metrics(reader, reference_positions, run, required_ids, True) for run in controls_positions
    ]
    candidates_metrics = [
        collect_run_metrics(reader, reference_positions, run, required_ids, True) for run in candidates_positions
    ]
    scopes = scope_definitions(gate, reference_positions)
    continuous_config = gate["noninferiority_rule"]["continuous"]
    factor = float(continuous_config["relative_noninferiority_factor"])
    floors = continuous_config["absolute_floors"]

    metric_results: dict[str, Any] = {"logits": {}, "final_hidden": {}, "layer_hidden": {}}
    # Near-margin top-1 swaps are evaluated only after every continuous gate
    # (including every layer) and every raw top-10 gate is known.  Doing this
    # in two passes avoids accidentally allowing a swap because an unrelated
    # scope has not been evaluated yet.
    all_continuous_pass = True
    for scope_name, scope_ids in scopes.items():
        if not scope_ids:
            continue
        scope_result: dict[str, Any] = {}
        for tensor_key, target in (("logits", "logits"), ("final_hidden", "final_hidden")):
            summaries_control = [aggregate_continuous(metrics, scope_ids, tensor_key, tensor_key == "logits") for metrics in controls_metrics]
            summaries_candidate = [aggregate_continuous(metrics, scope_ids, tensor_key, tensor_key == "logits") for metrics in candidates_metrics]
            aggregate_controls = [
                aggregate_metric_relative_l2(metrics, scope_ids, tensor_key) for metrics in controls_metrics
            ]
            aggregate_candidates = [
                aggregate_metric_relative_l2(metrics, scope_ids, tensor_key) for metrics in candidates_metrics
            ]
            max_scale = max(summary["reference_scale"] for summary in [*summaries_control, *summaries_candidate])
            metrics_gate: dict[str, Any] = {
                "aggregate_relative_l2": as_upper_gate(
                    aggregate_controls,
                    aggregate_candidates,
                    factor,
                    float(floors["relative_l2"]),
                ),
                "p99_position_relative_l2": as_upper_gate(
                    [summary["p99_position_relative_l2"] for summary in summaries_control],
                    [summary["p99_position_relative_l2"] for summary in summaries_candidate],
                    factor,
                    float(floors["relative_l2"]),
                ),
                "max_abs": as_upper_gate(
                    [summary["max_abs"] for summary in summaries_control],
                    [summary["max_abs"] for summary in summaries_candidate],
                    factor,
                    16.0 * ulp_f32(max_scale),
                ),
                "locations": {
                    "control": [
                        {
                            "p99_position_relative_l2": summary["p99_position_relative_l2_position"],
                            "max_abs": summary["max_abs_position"],
                        }
                        for summary in summaries_control
                    ],
                    "candidate": [
                        {
                            "p99_position_relative_l2": summary["p99_position_relative_l2_position"],
                            "max_abs": summary["max_abs_position"],
                        }
                        for summary in summaries_candidate
                    ],
                },
            }
            if tensor_key == "logits":
                metrics_gate["mean_kl_nats"] = as_upper_gate(
                    [summary["mean_kl_nats"] for summary in summaries_control],
                    [summary["mean_kl_nats"] for summary in summaries_candidate],
                    factor,
                    float(floors["mean_kl_nats"]),
                )
                metrics_gate["p99_kl_nats"] = as_upper_gate(
                    [summary["p99_kl_nats"] for summary in summaries_control],
                    [summary["p99_kl_nats"] for summary in summaries_candidate],
                    factor,
                    float(floors["p99_kl_nats"]),
                )
                metrics_gate["locations"]["control_kl"] = [summary["p99_kl_nats_position"] for summary in summaries_control]
                metrics_gate["locations"]["candidate_kl"] = [summary["p99_kl_nats_position"] for summary in summaries_candidate]
            metric_results[target][scope_name] = metrics_gate
            for gate_value in metrics_gate.values():
                if isinstance(gate_value, dict) and "passed" in gate_value:
                    all_continuous_pass &= bool(gate_value["passed"])

        discrete = evaluate_discrete_scope(
            gate,
            scope_ids,
            controls_metrics,
            candidates_metrics,
            False,
        )
        metric_results["logits"][scope_name]["discrete"] = discrete

    layer_ids = sorted(position_id for position_id, record in reference_positions.items() if record.get("layer_required"))
    for layer_index in range(LAYER_COUNT):
        if not layer_ids:
            break
        summaries_control: list[dict[str, Any]] = []
        summaries_candidate: list[dict[str, Any]] = []
        layer_metrics_control: list[dict[str, dict[str, VectorMetrics]]] = []
        layer_metrics_candidate: list[dict[str, dict[str, VectorMetrics]]] = []
        for metrics in controls_metrics:
            per_position = {
                position_id: {"layer": metrics[position_id]["layers"][layer_index]}
                for position_id in layer_ids
            }
            summaries_control.append(aggregate_continuous(per_position, layer_ids, "layer", False))
            layer_metrics_control.append(per_position)
        for metrics in candidates_metrics:
            per_position = {
                position_id: {"layer": metrics[position_id]["layers"][layer_index]}
                for position_id in layer_ids
            }
            summaries_candidate.append(aggregate_continuous(per_position, layer_ids, "layer", False))
            layer_metrics_candidate.append(per_position)
        aggregate_controls = [
            aggregate_metric_relative_l2(metrics, layer_ids, "layer") for metrics in layer_metrics_control
        ]
        aggregate_candidates = [
            aggregate_metric_relative_l2(metrics, layer_ids, "layer") for metrics in layer_metrics_candidate
        ]
        max_scale = max(summary["reference_scale"] for summary in [*summaries_control, *summaries_candidate])
        layer_result = {
            "aggregate_relative_l2": as_upper_gate(aggregate_controls, aggregate_candidates, factor, float(floors["relative_l2"])),
            "p99_position_relative_l2": as_upper_gate(
                [summary["p99_position_relative_l2"] for summary in summaries_control],
                [summary["p99_position_relative_l2"] for summary in summaries_candidate],
                factor,
                float(floors["relative_l2"]),
            ),
            "max_abs": as_upper_gate(
                [summary["max_abs"] for summary in summaries_control],
                [summary["max_abs"] for summary in summaries_candidate],
                factor,
                16.0 * ulp_f32(max_scale),
            ),
            "locations": {
                "control": [
                    {"p99_position_relative_l2": summary["p99_position_relative_l2_position"], "max_abs": summary["max_abs_position"]}
                    for summary in summaries_control
                ],
                "candidate": [
                    {"p99_position_relative_l2": summary["p99_position_relative_l2_position"], "max_abs": summary["max_abs_position"]}
                    for summary in summaries_candidate
                ],
            },
        }
        metric_results["layer_hidden"][f"layer_{layer_index:02}"] = layer_result
        for gate_value in layer_result.values():
            if isinstance(gate_value, dict) and "passed" in gate_value:
                all_continuous_pass &= bool(gate_value["passed"])

    # The reference's ``final-hidden.f32le`` is the post-final-norm hidden
    # state.  It has its own all-scope gate above and is also the explicit
    # ``final-norm hidden state`` member of the per-layer probe requirement.
    if layer_ids:
        final_norm_control = [
            aggregate_continuous(metrics, layer_ids, "final_hidden", False)
            for metrics in controls_metrics
        ]
        final_norm_candidate = [
            aggregate_continuous(metrics, layer_ids, "final_hidden", False)
            for metrics in candidates_metrics
        ]
        final_norm_aggregate_control = [
            aggregate_metric_relative_l2(metrics, layer_ids, "final_hidden") for metrics in controls_metrics
        ]
        final_norm_aggregate_candidate = [
            aggregate_metric_relative_l2(metrics, layer_ids, "final_hidden") for metrics in candidates_metrics
        ]
        final_norm_scale = max(
            summary["reference_scale"] for summary in [*final_norm_control, *final_norm_candidate]
        )
        final_norm_result = {
            "aggregate_relative_l2": as_upper_gate(
                final_norm_aggregate_control,
                final_norm_aggregate_candidate,
                factor,
                float(floors["relative_l2"]),
            ),
            "p99_position_relative_l2": as_upper_gate(
                [summary["p99_position_relative_l2"] for summary in final_norm_control],
                [summary["p99_position_relative_l2"] for summary in final_norm_candidate],
                factor,
                float(floors["relative_l2"]),
            ),
            "max_abs": as_upper_gate(
                [summary["max_abs"] for summary in final_norm_control],
                [summary["max_abs"] for summary in final_norm_candidate],
                factor,
                16.0 * ulp_f32(final_norm_scale),
            ),
            "locations": {
                "control": [
                    {
                        "p99_position_relative_l2": summary["p99_position_relative_l2_position"],
                        "max_abs": summary["max_abs_position"],
                    }
                    for summary in final_norm_control
                ],
                "candidate": [
                    {
                        "p99_position_relative_l2": summary["p99_position_relative_l2_position"],
                        "max_abs": summary["max_abs_position"],
                    }
                    for summary in final_norm_candidate
                ],
            },
        }
        metric_results["layer_hidden"]["final_norm"] = final_norm_result
        for gate_value in final_norm_result.values():
            if isinstance(gate_value, dict) and "passed" in gate_value:
                all_continuous_pass &= bool(gate_value["passed"])

    # Re-evaluate discrete gates after all continuous gates are known, as a
    # near-margin top-1 swap is admissible only when those gates and top-10
    # already pass.
    all_top10_pass = all(
        bool(scope_metrics["discrete"]["top10_wilson"]["passed"])
        for scope_metrics in metric_results["logits"].values()
        if "discrete" in scope_metrics
    )
    near_margin_allowed = all_continuous_pass and all_top10_pass
    discrete_results: dict[str, Any] = {}
    for scope_name, scope_ids in scopes.items():
        if scope_ids:
            discrete_results[scope_name] = evaluate_discrete_scope(
                gate,
                scope_ids,
                controls_metrics,
                candidates_metrics,
                near_margin_allowed,
            )
            metric_results["logits"][scope_name]["discrete"] = discrete_results[scope_name]

    primary_ids = scopes["aggregate_primary_decode"]
    bootstrap: list[dict[str, Any]] = []
    if len(primary_ids) == int(gate["corpus"]["primary_decode_position_count"]):
        # Select the scalar-median control run deterministically, then use it
        # for paired block resampling.  The JSON freezes a median baseline but
        # does not prescribe a tie rule for the per-position bootstrap pair.
        aggregate_discrete = discrete_results["aggregate_primary_decode"]
        control_rates = [sum(values) for values in aggregate_discrete["control_policy_agreement"]]
        ordered_controls = sorted(range(3), key=lambda index: (control_rates[index], index))
        median_control_index = ordered_controls[1]
        for candidate_values in aggregate_discrete["candidate_policy_agreement"]:
            bootstrap.append(
                bootstrap_lower_bound(
                    gate,
                    primary_ids,
                    reference_positions,
                    aggregate_discrete["control_policy_agreement"][median_control_index],
                    candidate_values,
                )
            )
    else:
        bootstrap.append({
            "applicable": False,
            "reason": "primary coverage incomplete; bootstrap is not evaluated in a test-only partial run",
        })

    failures: list[dict[str, Any]] = []
    for tensor_name, by_scope in metric_results.items():
        for scope_name, metrics in by_scope.items():
            for metric_name, metric in metrics.items():
                if isinstance(metric, dict) and "passed" in metric and not metric["passed"]:
                    failure = {
                        "tensor": tensor_name,
                        "scope": scope_name,
                        "metric": metric_name,
                        "detail": metric,
                    }
                    if isinstance(metrics.get("locations"), dict):
                        failure["locations"] = metrics["locations"]
                    failures.append(failure)
            if tensor_name == "logits" and "discrete" in metrics:
                discrete = metrics["discrete"]
                for name in ("top1_wilson", "top10_wilson"):
                    if not discrete[name]["passed"]:
                        failures.append(
                            {
                                "tensor": tensor_name,
                                "scope": scope_name,
                                "metric": name,
                                "detail": discrete[name],
                                "candidate_policy_mismatch_positions": discrete[
                                    "candidate_policy_mismatch_positions"
                                ],
                                "candidate_top10_mismatch_positions": discrete[
                                    "candidate_top10_mismatch_positions"
                                ],
                            }
                        )
                if discrete["hard_top1_regressions"]:
                    failures.append(
                        {
                            "tensor": tensor_name,
                            "scope": scope_name,
                            "metric": "hard_top1_regressions",
                            "detail": discrete["hard_top1_regressions"],
                        }
                    )
    for candidate_index, result in enumerate(bootstrap):
        if result.get("applicable", True) and not result.get("passed", False):
            failures.append({"tensor": "logits", "scope": "aggregate_primary_decode", "metric": f"block_bootstrap_candidate_{candidate_index}", "detail": result})

    test_only = not coverage["complete"] or bool(test_only_manifests)
    status = (
        "test_only_harness_verification"
        if test_only and not failures
        else "fail_relative_fp32_v0_2"
        if failures
        else "pass_relative_fp32_v0_2"
    )
    result = {
        "schema_version": RESULT_SCHEMA,
        "status": status,
        "frozen_gate_sha256": gate_sha,
        "candidate_id": next(iter(candidate_ids)),
        "capture_provenance": {
            "controls": [
                {
                    "capture_manifest": relative_path_or_absolute(path),
                    "selector": value.get("selector"),
                    "identity": value.get("identity"),
                }
                for path, value in zip(args.control, controls_values, strict=True)
            ],
            "candidates": [
                {
                    "capture_manifest": relative_path_or_absolute(path),
                    "candidate": value.get("candidate"),
                    "selector": value.get("selector"),
                    "identity": value.get("identity"),
                }
                for path, value in zip(args.candidate, candidates_values, strict=True)
            ],
        },
        "coverage": coverage,
        "scope_position_ids": scopes,
        "test_only_manifests": test_only_manifests,
        "identity_errors": [],
        "repeat_envelope": {
            "source": {
                "upper": gate["capture_contract"]["control"]["baseline_for_upper_error_metrics"],
                "lower": gate["capture_contract"]["control"]["baseline_for_lower_quality_metrics"],
            },
            "implemented": {
                "upper": "max(control_values) - median(control_values)",
                "lower": "median(control_values) - min(control_values)",
            },
        },
        "implementation_interpretations": [
            "P99 uses F64 nearest-rank ceil(0.99*n), because the frozen JSON fixes P99 but not an interpolation convention.",
            "Bootstrap seed is the first eight big-endian bytes of SHA-256(seed_domain UTF-8); RNG is NumPy PCG64, because the frozen JSON fixes the seed domain but not a PRNG/tie rule.",
            "Bootstrap pairs each candidate with the scalar-median control repetition; ties choose the lower repetition index.",
            "A hard candidate-only top-1 regression requires every control repetition to equal reference top-1, which is the conservative reading of 'control top-1 equals reference top-1'.",
            "max-abs ULP floor uses the maximum reference absolute value inside the evaluated tensor scope.",
        ],
        "metric_quantile_method": "nearest_rank_f64",
        "payload_hash_verification": not args.skip_payload_hash_verification,
        "metrics": metric_results,
        "bootstrap": bootstrap,
        "failures": failures,
    }
    write_json_new(args.output_json, result)
    write_markdown(args.output_markdown, result)
    return 0 if status in {"pass_relative_fp32_v0_2", "test_only_harness_verification"} else 1


def write_markdown(path: Path, result: Mapping[str, Any]) -> None:
    lines = [
        "# SQ8 numerical gate v0.2 consumer result",
        "",
        f"- Status: `{result.get('status')}`",
        f"- Frozen JSON SHA-256: `{result.get('frozen_gate_sha256')}`",
    ]
    coverage = result.get("coverage")
    if isinstance(coverage, dict):
        lines.extend(
            [
                f"- Complete frozen coverage: `{coverage.get('complete')}`",
                f"- Actual coverage: `{json.dumps(coverage.get('actual', {}), sort_keys=True)}`",
            ]
        )
    errors = result.get("identity_errors", [])
    if errors:
        lines.extend(["", "## Identity / input errors", ""])
        lines.extend(f"- {error}" for error in errors)
    failures = result.get("failures", [])
    if failures:
        lines.extend(["", "## Failing gates", ""])
        for failure in failures:
            context = {
                key: value
                for key, value in failure.items()
                if key not in {"tensor", "scope", "metric"}
            }
            lines.append(
                f"- `{failure.get('tensor')}` / `{failure.get('scope')}` / "
                f"`{failure.get('metric')}`: `{json.dumps(context, sort_keys=True)}`"
            )
    elif result.get("status") == "test_only_harness_verification":
        lines.extend(
            [
                "",
                "All measured metrics passed, but this is explicitly not an admission result because it uses incomplete coverage and/or test-only capture manifests.",
            ]
        )
    else:
        lines.extend(["", "All required measured gates passed."])
    interpretations = result.get("implementation_interpretations", [])
    if interpretations:
        lines.extend(["", "## Recorded interpretation details", ""])
        lines.extend(f"- {value}" for value in interpretations)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise GateError(f"refusing to overwrite existing output {path}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def clone_manifest(args: argparse.Namespace) -> int:
    """Create a test-only control/candidate manifest that aliases an index.

    This command is intentionally useful only for consumer self consistency;
    it cannot produce an admissible capture because its identity says
    ``test_only_reference_alias``.
    """

    index = load_json(args.reference)
    if index.get("schema_version") != REFERENCE_INDEX_SCHEMA:
        raise GateError("clone-manifest requires a reference index")
    role = args.role
    manifest = {
        "schema_version": CAPTURE_SCHEMA,
        "role": role,
        "frozen_gate": index["frozen_gate"],
        "candidate": {"id": args.candidate_id},
        "selector": {"enabled": role == "candidate", "kind": "test_only_reference_alias"},
        "identity": {
            "artifact_content_sha256": index["identity"]["artifact_content_sha256"],
            "fixture_manifest_sha256": index["identity"]["fixture_manifest_sha256"],
            "materialized_token_hashes": index["identity"]["materialized_token_hashes"],
            "reference_executable_sha256": index["identity"]["reference_executable_sha256"],
            "executable_sha256": "test_only_reference_alias",
            "selector_configuration_fingerprint": f"test-only:{role}:{args.candidate_id}",
            "device_identity": {"backend": "test_only_reference_alias"},
            "runtime_compiler_versions": {"backend": "test_only_reference_alias"},
            "hip_guard_environment": {"backend": "test_only_reference_alias"},
        },
        "positions": index["positions"],
        "test_only": True,
    }
    write_json_new(args.output, manifest)
    return 0


def mutate_one_logit(args: argparse.Namespace) -> int:
    """Make one deliberately broken candidate manifest for negative tests."""

    manifest = load_json(args.input)
    if manifest.get("schema_version") != CAPTURE_SCHEMA:
        raise GateError("mutate-one-logit requires a capture manifest")
    output_root = args.output.parent
    output_root.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise GateError(f"refusing to overwrite existing output {args.output}")
    copied = json.loads(json.dumps(manifest))
    position = copied["positions"][0]
    source = Path(position["logits"]["path"])
    target = output_root / "deliberately-corrupt-logits.f32le"
    if target.exists():
        raise GateError(f"refusing to overwrite existing corrupt tensor {target}")
    shutil.copyfile(source, target)
    values = np.memmap(target, dtype="<f4", mode="r+")
    values[0] = np.float32(values[0] + 100.0)
    values.flush()
    del values
    position["logits"] = tensor_descriptor(target, sha256_file(target), VOCAB_SIZE)
    copied["test_only_deliberately_corrupt"] = True
    write_json_new(args.output, copied)
    return 0


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    sub = value.add_subparsers(dest="command", required=True)

    index = sub.add_parser("index-reference", help="read-only index of the CPU reference capture")
    index.add_argument("--gate", type=Path, required=True)
    index.add_argument("--reference-root", type=Path, required=True)
    index.add_argument("--output", type=Path, required=True)
    index.set_defaults(function=index_reference)

    evaluate_parser = sub.add_parser("evaluate", help="evaluate controls and one candidate")
    evaluate_parser.add_argument("--gate", type=Path, required=True)
    evaluate_parser.add_argument("--reference", type=Path, required=True)
    evaluate_parser.add_argument("--control", type=Path, action="append", required=True)
    evaluate_parser.add_argument("--candidate", type=Path, action="append", required=True)
    evaluate_parser.add_argument("--output-json", type=Path, required=True)
    evaluate_parser.add_argument("--output-markdown", type=Path, required=True)
    evaluate_parser.add_argument("--allow-incomplete-test-coverage", action="store_true")
    evaluate_parser.add_argument("--skip-payload-hash-verification", action="store_true")
    evaluate_parser.set_defaults(function=evaluate)

    clone = sub.add_parser("clone-manifest", help="test-only reference alias manifest")
    clone.add_argument("--reference", type=Path, required=True)
    clone.add_argument("--role", choices=("control", "candidate"), required=True)
    clone.add_argument("--candidate-id", required=True)
    clone.add_argument("--output", type=Path, required=True)
    clone.set_defaults(function=clone_manifest)

    mutate = sub.add_parser("mutate-one-logit", help="create a deliberate negative-test candidate")
    mutate.add_argument("--input", type=Path, required=True)
    mutate.add_argument("--output", type=Path, required=True)
    mutate.set_defaults(function=mutate_one_logit)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        return int(args.function(args))
    except GateError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
