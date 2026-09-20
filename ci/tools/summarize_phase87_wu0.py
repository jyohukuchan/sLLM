#!/usr/bin/env python3
"""Summarize the Phase 87 WU0 read-only bandwidth measurements.

The input is deliberately narrow: two exact-target probe directories, each
with three copy and three read JSONL results for the frozen 73-case payload
list.  This is an evidence joiner, not a generic benchmark framework.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any


TARGETS = ("gfx1030", "gfx1201")
ROUNDS = (0, 1, 2)
TRANSITIONS = 127
READ_SCHEMA = "phase87-read-bandwidth-v1"
COPY_SCHEMA = "phase87-copy-bandwidth-v1"
EXEC_SCHEMA = "phase87-wu0-execution-v1"


class SummaryError(RuntimeError):
    pass


def fail(message: str) -> None:
    raise SummaryError(message)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_json(path: Path) -> Any:
    if not path.is_file():
        fail(f"missing JSON: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        fail(f"invalid JSON {path}: {error}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        fail(f"missing JSONL: {path}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            fail(f"invalid JSONL {path}:{line_number}: {error}")
        if not isinstance(value, dict):
            fail(f"JSONL row is not an object: {path}:{line_number}")
        rows.append(value)
    return rows


def finite_positive(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        fail(f"{label} is not numeric: {value!r}")
    if not math.isfinite(number) or number <= 0.0:
        fail(f"{label} is not finite and positive: {value!r}")
    return number


def key(row: dict[str, Any]) -> tuple[int, int, int, str]:
    try:
        return (int(row["m"]), int(row["k"]), int(row["n"]), str(row["encoding"]))
    except (KeyError, TypeError, ValueError) as error:
        fail(f"invalid payload key: {error}")
    raise AssertionError


def decimal_gbps(row: dict[str, Any]) -> float:
    return finite_positive(row["effective_gib_per_s"], "effective_gib_per_s") * (1024.0**3) / 1e9


def median_range(values: list[float]) -> dict[str, float]:
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def validate_execution(directory: Path, target: str) -> dict[str, Any]:
    path = directory / "execution.json"
    execution = read_json(path)
    if execution.get("schema") != EXEC_SCHEMA or execution.get("state") != "complete":
        fail(f"execution is not complete v1: {path}")
    if execution.get("target") != target:
        fail(f"execution target mismatch: {path}")
    for path_field, hash_field in (("binary", "binary_sha256"), ("payloads", "payloads_sha256")):
        source = Path(execution[path_field])
        if digest(source) != execution.get(hash_field):
            fail(f"execution {path_field} hash differs: {source}")
    jobs = execution.get("jobs")
    expected = {f"{mode}-{round_number}" for round_number in ROUNDS for mode in ("copy", "read")}
    if not isinstance(jobs, list) or {job.get("name") for job in jobs} != expected or len(jobs) != 6:
        fail(f"execution does not contain exactly 3 copy/read rounds: {path}")
    hashes: dict[str, str] = {"execution": digest(path)}
    for job in jobs:
        name = str(job["name"])
        mode = str(job["mode"])
        if mode not in ("copy", "read") or int(job.get("round", -1)) not in ROUNDS:
            fail(f"invalid execution job identity: {path} {name}")
        command = [str(value) for value in job.get("command", [])]
        if "--warmups" not in command or command[command.index("--warmups") + 1] != "3":
            fail(f"job {name} does not use three warmups")
        if "--measured" not in command or command[command.index("--measured") + 1] != "9":
            fail(f"job {name} does not use nine measured samples")
        if directory.name.startswith("warm-") and (
                "--warmup-ms" not in command or command[command.index("--warmup-ms") + 1] != "300"):
            fail(f"job {name} does not use the 300 ms continuous warmup")
        if int(job.get("case_count", -1)) != 73 or int(job.get("exit_code", -1)) != 0:
            fail(f"job {name} did not report 73 successful cases")
        output = directory / f"{name}.jsonl"
        observed_hash = digest(output)
        if job.get("output_sha256") != observed_hash:
            fail(f"execution output hash mismatch for {output}")
        hashes[name] = observed_hash
    if execution.get("performance_level_restored") is not True:
        fail(f"performance level was not restored: {path}")
    if target == "gfx1201" and execution.get("service_restored") is not True:
        fail(f"R9700 service was not restored: {path}")
    return {
        "path": str(path),
        "sha256": digest(path),
        "target": target,
        "binary_sha256": execution.get("binary_sha256"),
        "payloads_sha256": execution.get("payloads_sha256"),
        "restoration": {
            "performance_level_restored": execution.get("performance_level_restored"),
            "service_restored": execution.get("service_restored") if target == "gfx1201" else None,
        },
        "output_sha256": hashes,
    }


def validate_mode(directory: Path, target: str, mode: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows_by_round: list[list[dict[str, Any]]] = []
    hashes: dict[str, str] = {}
    expected_keys: list[tuple[int, int, int, str]] | None = None
    expected_payloads: list[int] | None = None
    for round_number in ROUNDS:
        path = directory / f"{mode}-{round_number}.jsonl"
        rows = read_jsonl(path)
        if len(rows) != 73:
            fail(f"{path} contains {len(rows)} rows, expected 73")
        hashes[str(round_number)] = digest(path)
        keys = [key(row) for row in rows]
        payloads = [int(row.get("payload_bytes", -1)) for row in rows]
        if expected_keys is None:
            expected_keys, expected_payloads = keys, payloads
        if keys != expected_keys or payloads != expected_payloads:
            fail(f"{mode} payload order or byte size changed in {path}")
        for row in rows:
            if row.get("status") != "PASS" or row.get("target") != target:
                fail(f"{path} contains a non-PASS or wrong-target row")
            if mode == "read":
                if row.get("schema_version") != READ_SCHEMA or row.get("mode") != "read":
                    fail(f"read schema/mode mismatch in {path}")
                if int(row.get("scalar_output_bytes", -1)) != 4 * int(row.get("read_grid_blocks", -1)):
                    fail(f"read scalar output contract mismatch in {path}")
                if int(row.get("read_grid_blocks", 0)) <= 0:
                    fail(f"zero read grid in {path}")
                if int(row.get("logical_read_write_bytes", -1)) != int(row["payload_bytes"]) + int(row["scalar_output_bytes"]):
                    fail(f"read logical read/write contract mismatch in {path}")
                if int(row["payload_bytes"]) >= 1024 * 1024 and int(row.get("source_working_set_bytes", 0)) < 512 * 1024 * 1024:
                    fail(f"read working set does not exceed 512 MiB in {path}")
            else:
                if row.get("schema_version") != COPY_SCHEMA:
                    fail(f"copy schema mismatch in {path}")
                if int(row.get("copy_traffic_bytes", -1)) != 2 * int(row["payload_bytes"]):
                    fail(f"copy traffic contract mismatch in {path}")
                if int(row["payload_bytes"]) >= 1024 * 1024 and int(row.get("working_set_bytes", 0)) < 512 * 1024 * 1024:
                    fail(f"copy working set does not exceed 512 MiB in {path}")
            median_ms = finite_positive(row.get("median_ms"), f"{path} median_ms")
            if "samples_ms" in row:
                samples = row["samples_ms"]
                if not isinstance(samples, list) or len(samples) != 9:
                    fail(f"{mode} samples_ms must contain nine measured samples in {path}")
                sample_values = [finite_positive(value, f"{path} samples_ms") for value in samples]
                if not math.isclose(statistics.median(sample_values), median_ms, rel_tol=3e-5, abs_tol=1e-6):
                    fail(f"{mode} samples_ms median mismatch in {path}")
            measured_gib = finite_positive(row["effective_gib_per_s"], "effective_gib_per_s")
            denominator = int(row["payload_bytes"]) if mode == "read" else int(row["copy_traffic_bytes"])
            expected = denominator / (median_ms / 1000.0) / 2.0**30
            if not math.isclose(measured_gib, expected, rel_tol=3e-5, abs_tol=1e-6):
                fail(f"{mode} bandwidth arithmetic mismatch in {path}: {measured_gib} != {expected}")
        rows_by_round.append(rows)
    assert expected_keys is not None and expected_payloads is not None
    return rows_by_round[0], {"hashes": hashes, "keys": expected_keys, "payloads": expected_payloads, "rounds": rows_by_round}


def load_probe(root: Path, variant: str, target: str) -> dict[str, Any]:
    directory = root / f"{variant}-{target}"
    if not directory.is_dir():
        fail(f"missing {variant} probe directory: {directory}")
    execution = validate_execution(directory, target)
    copy_first, copy_meta = validate_mode(directory, target, "copy")
    read_first, read_meta = validate_mode(directory, target, "read")
    if copy_meta["keys"] != read_meta["keys"] or copy_meta["payloads"] != read_meta["payloads"]:
        fail(f"copy/read payload sets differ for {target}")
    cases: list[dict[str, Any]] = []
    for index, payload_key in enumerate(copy_meta["keys"]):
        copy_values = [decimal_gbps(round_rows[index]) for round_rows in copy_meta["rounds"]]
        read_values = [decimal_gbps(round_rows[index]) for round_rows in read_meta["rounds"]]
        copy_row, read_row = copy_first[index], read_first[index]
        cases.append({
            "key": list(payload_key),
            "m": payload_key[0], "k": payload_key[1], "n": payload_key[2], "encoding": payload_key[3],
            "payload_bytes": int(copy_row["payload_bytes"]),
            "latency_cache_sensitive": int(copy_row["payload_bytes"]) < 1024**2,
            "read_scalar_output_bytes": int(read_row["scalar_output_bytes"]),
            "read_source_working_set_bytes": int(read_row["source_working_set_bytes"]),
            "copy_gbps": median_range(copy_values),
            "read_gbps": median_range(read_values),
            "copy_median_ms": statistics.median(float(rows[index]["median_ms"]) for rows in copy_meta["rounds"]),
            "read_median_ms": statistics.median(float(rows[index]["median_ms"]) for rows in read_meta["rounds"]),
        })
    return {"target": target, "execution": execution, "cases": cases, "by_key": {tuple(case["key"]): case for case in cases}}


def weight_bytes(encoding: str, k: int, n: int) -> int:
    if encoding == "bf16":
        return 2 * k * n
    if encoding == "fp8":
        return k * n + 4 * n
    if encoding == "nvfp4":
        return n * ((k + 1) // 2) + n * ((k + 15) // 16) + 4
    fail(f"unsupported weight format: {encoding}")
    raise AssertionError


def stage0_read_match(row: dict[str, Any], by_key: dict[tuple[int, int, int, str], dict[str, Any]]) -> dict[str, Any]:
    candidates = row.get("shape_candidates") or []
    matches: list[dict[str, Any]] = []
    candidate_keys: list[list[Any]] = []
    for candidate in candidates:
        candidate_key = (int(candidate["m"]), int(candidate["k"]), int(candidate["n"]), str(candidate["encoding"]))
        if candidate_key[3] == "bytes":
            candidate_key = (1, 2 * candidate_key[0] * candidate_key[1], 1, "bytes")
        candidate_keys.append(list(candidate_key))
        if candidate_key in by_key:
            matches.append(by_key[candidate_key])
    copy_reference = row.get("copy_reference") or {}
    if not matches and copy_reference.get("reference_rows"):
        for reference in copy_reference["reference_rows"]:
            payload = int(reference["payload_bytes"])
            matches.extend(case for case in by_key.values() if case["encoding"] == "bytes" and case["payload_bytes"] == payload)
    read_rates = [case["read_gbps"]["median"] for case in matches]
    return {
        "candidate_keys": candidate_keys,
        "matched_payloads": [case["payload_bytes"] for case in matches],
        "read_gbps": median_range(read_rates) if read_rates else None,
        "read_cases": matches,
        "match_kind": "exact_shape" if len(candidates) == 1 and len(matches) == 1 else ("candidate_range" if matches else "unavailable"),
    }


def headroom_for_row(pair: dict[str, Any], row: dict[str, Any], by_key: dict[tuple[int, int, int, str], dict[str, Any]]) -> dict[str, Any] | None:
    if row.get("family") not in ("fp8_matmul", "nvfp4_matmul"):
        return None
    candidates = row.get("shape_candidates") or []
    if not candidates:
        return None
    calls = float(row["invocation_count"]) / float(pair["committed_decode_transitions"])
    observed = float(row["duration_ms_per_transition"])
    options: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_key = (int(candidate["m"]), int(candidate["k"]), int(candidate["n"]), str(candidate["encoding"]))
        probe = by_key.get(candidate_key)
        if probe is None:
            continue
        weight = weight_bytes(candidate_key[3], candidate_key[1], candidate_key[2])
        floor = weight * calls / (probe["read_gbps"]["median"] * 1e6)
        options.append({"shape": list(candidate_key), "weight_bytes": weight, "read_gbps": probe["read_gbps"], "floor_ms": floor, "signed_ms": observed - floor, "positive_ms": max(0.0, observed - floor)})
        options[-1]["activation_bytes"] = probe["payload_bytes"] - weight
    if not options:
        return {"state": "unavailable", "observed_ms": observed, "calls_per_token": calls, "candidate_keys": [list((int(c["m"]), int(c["k"]), int(c["n"]), str(c["encoding"]))) for c in candidates]}
    floors = [item["floor_ms"] for item in options]
    signed = [item["signed_ms"] for item in options]
    positive = [item["positive_ms"] for item in options]
    result = {
        "state": "exact" if len(options) == 1 else "range",
        "calls_per_token": calls,
        "observed_ms_per_token": observed,
        "candidates": options,
        "time_floor_ms_per_token": floors[0] if len(floors) == 1 else {"min": min(floors), "max": max(floors)},
        "headroom_signed_ms": signed[0] if len(signed) == 1 else {"min": min(signed), "max": max(signed)},
        "headroom_positive_ms": positive[0] if len(positive) == 1 else {"min": min(positive), "max": max(positive)},
        "caveat": "read-only payload floor; not a hard physical DRAM bound",
        "probe_slower_than_current_kernel": any(value < 0 for value in signed),
    }
    if pair["mtp"]:
        result["state"] = "range" if len(options) > 1 else "mtp_shape_bound"
        result["caveat"] += "; MTP mixed dispatch, no invented exact shape"
    return result


def aggregate_headroom(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for row in rows:
        family = row.get("family")
        budget = row.get("headroom")
        if family not in ("fp8_matmul", "nvfp4_matmul") or not budget or budget.get("state") != "exact":
            continue
        item = result.setdefault(family, {"observed_ms_per_token": 0.0, "floor_ms_per_token": 0.0, "headroom_signed_ms": 0.0, "headroom_positive_ms": 0.0, "rows": 0})
        item["observed_ms_per_token"] += budget["observed_ms_per_token"]
        item["floor_ms_per_token"] += budget["time_floor_ms_per_token"]
        item["headroom_signed_ms"] += budget["headroom_signed_ms"]
        item["headroom_positive_ms"] += budget["headroom_positive_ms"]
        item["rows"] += 1
    for item in result.values():
        item["cutoff_half_positive_ms"] = item["headroom_positive_ms"] / 2.0
        item["cutoff_applicable"] = item["headroom_positive_ms"] > 0
        if not item["cutoff_applicable"]:
            item["note"] = "Probe reference does not establish positive headroom; zero is not proof that optimization is impossible or a new zero-gain gate."
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--variant", choices=("v2", "v3", "v3-timed", "pipeline", "warm"), required=True)
    parser.add_argument("--stage0", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        root = args.root.resolve()
        stage0_path = args.stage0.resolve()
        reuse_path = root / "baseline-reuse.json"
        reuse = read_json(reuse_path)
        if reuse.get("binary_matches") != {target: True for target in TARGETS} or reuse.get("route_sources_unchanged") is not True:
            fail("baseline-reuse does not prove unchanged inference binaries and routes")
        probes = {target: load_probe(root, args.variant, target) for target in TARGETS}
        stage0 = read_json(stage0_path)
        pairs = stage0.get("pairs")
        if not isinstance(pairs, list) or len(pairs) != 4:
            fail("stage0 result does not contain four pairs")
        row_outputs: list[dict[str, Any]] = []
        mtp_outputs: list[dict[str, Any]] = []
        for pair in pairs:
            target = str(pair.get("target"))
            if target not in TARGETS or not isinstance(pair.get("rows"), list):
                fail("invalid stage0 pair")
            by_key = probes[target]["by_key"]
            output_rows: list[dict[str, Any]] = []
            for row in pair["rows"]:
                item = {"kernel_name": row.get("kernel_name"), "grid_x": row.get("grid_x"), "family": row.get("family"), "priority": row.get("priority"), "stage0_mtp": pair.get("mtp"), "stage0_duration_ms_per_token": row.get("duration_ms_per_transition"), "stage0_calls_per_token": float(row.get("invocation_count", 0)) / float(pair.get("committed_decode_transitions", TRANSITIONS)), "copy_reference": row.get("copy_reference"), "read_match": stage0_read_match(row, by_key)}
                budget = headroom_for_row(pair, row, by_key)
                if budget is not None:
                    item["headroom"] = budget
                output_rows.append(item)
            (mtp_outputs if pair["mtp"] else row_outputs).append({"target": target, "mtp": bool(pair["mtp"]), "rows": output_rows})
        primary = {
            "mtp_off_by_target": {target: aggregate_headroom([item for pair in row_outputs if pair["target"] == target for item in pair["rows"]]) for target in TARGETS},
            "cutoff_definition": "half of clipped-positive summed headroom; signed values remain visible",
            "wu2_r9700_fp8": {**aggregate_headroom([item for pair in row_outputs if pair["target"] == "gfx1201" for item in pair["rows"]]).get("fp8_matmul", {}), "target": "gfx1201", "scope": "MTP-off"},
        }
        attention = next((pair for pair in pairs if pair.get("target") == "gfx1030" and not pair.get("mtp") and any("stage1_kernel" in str(row.get("kernel_name")) for row in pair.get("rows", []))), None)
        if attention is not None:
            stage1 = next(row for row in attention["rows"] if "stage1_kernel" in str(row.get("kernel_name")))
            context_rows = {case["payload_bytes"]: case for case in probes["gfx1030"]["cases"] if case["encoding"] == "bytes" and case["payload_bytes"] in (17315904, 17448960, 17582016)}
            ranges = []
            for context, payload in ((8193, 17315904), (8256, 17448960), (8319, 17582016)):
                case = context_rows.get(payload)
                if case is None:
                    continue
                kv_bytes = 2112 * context
                floor = kv_bytes * (float(stage1["invocation_count"]) / float(attention["committed_decode_transitions"])) / (case["read_gbps"]["median"] * 1e6)
                ranges.append({"context_tokens": context, "unique_kv_bytes_per_dispatch": kv_bytes, "read_payload_bytes": payload, "read_gbps": case["read_gbps"], "floor_ms_per_token": floor, "headroom_signed_ms": float(stage1["duration_ms_per_transition"]) - floor, "headroom_positive_ms": max(0.0, float(stage1["duration_ms_per_transition"]) - floor)})
            primary["wu1_v620_attention_stage1"] = {"target": "gfx1030", "scope": "MTP-off", "observed_ms_per_token": stage1["duration_ms_per_transition"], "calls_per_token": float(stage1["invocation_count"]) / float(attention["committed_decode_transitions"]), "context_ranges": ranges, "caveat": "unique K/V payload; query bytes are included in read probe denominator; read-only floor is not a hard physical ceiling"}
            central = next(row for row in ranges if row["context_tokens"] == 8256)
            primary["wu1_v620_attention_stage1"].update(
                representative_context_tokens=8256,
                headroom_positive_ms=central["headroom_positive_ms"],
                cutoff_half_positive_ms=central["headroom_positive_ms"] / 2,
            )
        secondary_pair = next(pair for pair in pairs if pair["target"] == "gfx1201" and not pair["mtp"])
        secondary_row = next(row for row in secondary_pair["rows"] if "stage1_kernel" in row["kernel_name"])
        secondary_rate = probes["gfx1201"]["by_key"][(1, 17448960, 1, "bytes")]["read_gbps"]
        secondary_calls = secondary_row["invocation_count"] / secondary_pair["committed_decode_transitions"]
        secondary_floor = (2112 * 8256 * secondary_calls) / (secondary_rate["median"] * 1e6)
        secondary_margin = secondary_row["duration_ms_per_transition"] - secondary_floor
        primary["wu1_r9700_secondary_attention_stage1"] = {
            "target": "gfx1201", "scope": "MTP-off", "representative_context_tokens": 8256,
            "observed_ms_per_token": secondary_row["duration_ms_per_transition"],
            "read_gbps": secondary_rate, "floor_ms_per_token": secondary_floor,
            "headroom_signed_ms": secondary_margin,
            "headroom_positive_ms": max(0, secondary_margin),
            "cutoff_half_positive_ms": max(0, secondary_margin) / 2,
        }
        probe_output = {
            target: {name: value for name, value in probe.items() if name != "by_key"}
            for target, probe in probes.items()
        }
        source_refs: dict[str, Any] = {"runner": str(Path(__file__).resolve()), "runner_sha256": digest(Path(__file__).resolve())}
        for name in (f"build-{args.variant}-gfx1030.log", f"build-{args.variant}-gfx1201.log", "read-probe-v1.hip.cpp", "read-probe-v2.hip.cpp", "read-probe-v3-before-timing-fix.hip.cpp", "read-probe-final.hip.cpp", "build-identity.json", "isa-check.json", "independent-checksum-audit.json"):
            path = root / name
            if path.is_file():
                source_refs[name] = {"path": str(path), "sha256": digest(path)}
        caveats = [
            "Read-only bandwidth is an observed reference, not a hard physical DRAM ceiling. Kernel/reduction cost and access/cache differences can make actual matmul logical bandwidth higher.",
            "MTP mixed-shape rows retain candidate ranges and do not claim a fabricated exact shape.",
            "Small payloads are launch/latency dominated; their read rate is tagged by payload size in the case rows.",
        ]
        if args.variant == "warm":
            caveats.append("The warm variant adds 300 ms of continuous unsynchronized launches before each payload's measured samples, so automatic clocks match a continuously running decoder. The earlier pipeline variant measured partly before clocks settled and understated R9700 mid-size payload rates by about 10%.")
        if args.variant not in ("v3-timed", "pipeline", "warm"):
            caveats.append("This probe variant predates the final separated-timing protocol; read rates can be conservative due to reduction/D2H timing effects.")
        output = {
            "schema": "phase87-wu0-summary-v1",
            "state": "PASS",
            "variant": args.variant,
            "scope": "Phase 87 WU0 exact-target read-only/copy bandwidth; MTP-off primary cutoffs",
            "domains": {"read_gbps": "decimal GB/s from payload bytes / measured duration", "copy_gbps": "decimal GB/s from copy read+write traffic", "counter": "not used; GL2C/EA request bytes are not logical payload"},
            "formulas": {"bf16_weight_bytes": "2*K*N", "fp8_weight_bytes": "K*N+4*N", "nvfp4_weight_bytes": "N*ceil(K/2)+N*ceil(K/16)+4", "time_floor_ms": "logical_bytes_per_token/(read_gbps*1e6)", "signed_headroom_ms": "observed_ms-time_floor_ms", "positive_headroom_ms": "max(0,signed_headroom_ms)"},
            "baseline_reuse": {"path": str(reuse_path), "sha256": digest(reuse_path), "value": reuse},
            "stage0": {"path": str(stage0_path), "sha256": digest(stage0_path)},
            "sources": source_refs,
            "probes": probe_output,
            "primary": primary,
            "stage0_rows_mtp_off": row_outputs,
            "stage0_rows_mtp_mixed": mtp_outputs,
            "caveats": caveats,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"PASS {args.output}")
        return 0
    except SummaryError as error:
        print(f"phase87 WU0 summary: FAIL-CLOSED: {error}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
