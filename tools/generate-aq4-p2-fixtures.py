#!/usr/bin/env python3
"""Generate deterministic, hash-bound AQ4 P2 prompt fixtures.

The generated prompt ids are synthetic benchmark inputs.  They are intentionally derived from the
expanded case identity instead of user text, so fixture/index artifacts contain no prompt content.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

MAX_JSON_BYTES = 32 * 1024 * 1024
MAX_FIXTURE_TOKENS = 4096
MAX_CASES = 65536
ID_RE = re.compile(r"^[A-Za-z0-9._:-]+$")
SUBSETS = ("smoke", "representative", "full", "all")
FIXTURE_SCHEMA = "ullm.aq4_p2_case_fixture.v1"
INDEX_SCHEMA = "ullm.aq4_p2_fixture_index.v1"
SUPPORTED_PREFILL_SCOPES = {"full_model"}


class FixtureError(ValueError):
    pass


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def reject_symlink_components(path: Path) -> None:
    absolute = path.absolute()
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        if current.is_symlink():
            raise FixtureError(f"symlink path component rejected: {current}")


def read_regular(path: Path, label: str, maximum: int = MAX_JSON_BYTES) -> tuple[dict[str, Any], bytes]:
    reject_symlink_components(path)
    if path.is_symlink() or not path.is_file():
        raise FixtureError(f"{label} must be a regular file")
    if path.stat().st_size > maximum:
        raise FixtureError(f"{label} exceeds bounded size")
    raw = path.read_bytes()
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_strict_pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(FixtureError(f"non-finite JSON number: {item}")),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise FixtureError(f"{label} JSON rejected: {error}") from error
    if not isinstance(value, dict):
        raise FixtureError(f"{label} root must be an object")
    return value, raw


def _strict_pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise FixtureError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def safe_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 256 or ID_RE.fullmatch(value) is None:
        raise FixtureError(f"invalid {label}")
    return value


def case_hash(case: dict[str, Any]) -> str:
    value = json.loads(json.dumps(case))
    value["case_sha256"] = None
    return sha256_bytes(canonical(value))


def file_sha256(path: Path, label: str, maximum: int = 512 * 1024 * 1024) -> str:
    reject_symlink_components(path)
    if path.is_symlink() or not path.is_file():
        raise FixtureError(f"{label} must be a regular file")
    if path.stat().st_size > maximum:
        raise FixtureError(f"{label} exceeds bounded size")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_served_model(manifest: dict[str, Any], root: Path) -> dict[str, Any]:
    if manifest.get("schema_version") != "ullm.served_model.v2":
        raise FixtureError("served model schema is not v2")
    public = manifest.get("public")
    generation = manifest.get("generation")
    tokenizer = manifest.get("tokenizer")
    if not isinstance(public, dict) or not isinstance(generation, dict) or not isinstance(tokenizer, dict):
        raise FixtureError("served model public/generation/tokenizer contract is incomplete")
    context_length = public.get("context_length")
    vocab_size = generation.get("vocab_size")
    max_completion = generation.get("max_completion_tokens")
    eos_ids = generation.get("eos_token_ids")
    if not isinstance(context_length, int) or not 1 <= context_length <= 4096:
        raise FixtureError("served context_length is outside the bounded contract")
    if not isinstance(vocab_size, int) or not 1 <= vocab_size <= 1_000_000:
        raise FixtureError("served vocab_size is outside the bounded contract")
    if not isinstance(max_completion, int) or not 0 <= max_completion <= 4096:
        raise FixtureError("served max_completion_tokens is outside the bounded contract")
    if not isinstance(eos_ids, list) or any(not isinstance(item, int) or not 0 <= item < vocab_size for item in eos_ids):
        raise FixtureError("served eos_token_ids are invalid")
    root_value = tokenizer.get("root")
    files = tokenizer.get("files")
    if not isinstance(root_value, str) or not isinstance(files, dict) or not files:
        raise FixtureError("served tokenizer file contract is incomplete")
    tokenizer_root = Path(root_value)
    if not tokenizer_root.is_absolute():
        tokenizer_root = root / tokenizer_root
    reject_symlink_components(tokenizer_root)
    reserved = set(eos_ids)
    reasoning = manifest.get("reasoning")
    if isinstance(reasoning, dict):
        for field in ("start_token_ids", "end_token_ids", "forced_end_token_ids"):
            values = reasoning.get(field, [])
            if not isinstance(values, list) or any(not isinstance(item, int) or not 0 <= item < vocab_size for item in values):
                raise FixtureError(f"served reasoning {field} is invalid")
            reserved.update(values)
    hashes: dict[str, str] = {}
    for name, expected in files.items():
        safe_id(name, "tokenizer file name")
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise FixtureError(f"tokenizer hash is invalid for {name}")
        actual = file_sha256(tokenizer_root / name, f"tokenizer/{name}", 128 * 1024 * 1024)
        if actual != expected:
            raise FixtureError(f"tokenizer hash differs for {name}")
        hashes[name] = actual
    return {
        "context_length": context_length,
        "vocab_size": vocab_size,
        "max_completion_tokens": max_completion,
        "eos_token_ids": sorted(set(eos_ids)),
        "reserved_token_ids": sorted(reserved),
        "reserved_token_ids_sha256": sha256_bytes(canonical(sorted(reserved))),
        "tokenizer_root": str(tokenizer_root),
        "tokenizer_files_sha256": hashes,
    }


def validate_expanded(expanded: dict[str, Any], served: dict[str, Any]) -> list[dict[str, Any]]:
    if expanded.get("schema_version") != "ullm.aq4_production_p2_expanded.v2":
        raise FixtureError("expanded manifest schema is not v2")
    cases = expanded.get("cases")
    if not isinstance(cases, list) or not 1 <= len(cases) <= MAX_CASES:
        raise FixtureError("expanded cases are outside bounded limits")
    if expanded.get("case_count") != len(cases):
        raise FixtureError("expanded case_count differs")
    seen: set[str] = set()
    for case in cases:
        if not isinstance(case, dict):
            raise FixtureError("expanded case is not an object")
        case_id = safe_id(case.get("case_id"), "case_id")
        if case_id in seen:
            raise FixtureError(f"duplicate expanded case_id: {case_id}")
        seen.add(case_id)
        if case.get("case_sha256") != case_hash(case):
            raise FixtureError(f"expanded case hash differs: {case_id}")
        prompt = case.get("prompt_tokens")
        context = case.get("context_tokens")
        generated = case.get("generated_tokens")
        prefix = case.get("cached_prefix_tokens")
        for value, label in ((prompt, "prompt_tokens"), (context, "context_tokens"), (generated, "generated_tokens"), (prefix, "cached_prefix_tokens")):
            if not isinstance(value, int) or value < 0:
                raise FixtureError(f"{case_id} has invalid {label}")
        if prompt < 1 or prompt > MAX_FIXTURE_TOKENS or context > served["context_length"] or prefix + prompt != context:
            raise FixtureError(f"{case_id} violates context contract")
        if generated > served["max_completion_tokens"] or context + generated > served["context_length"]:
            raise FixtureError(f"{case_id} exceeds generation/context contract")
    return cases


def token_ids(case: dict[str, Any], count: int, vocab_size: int, reserved: set[int]) -> list[int]:
    values: list[int] = []
    index = 0
    seed = f"{case['case_id']}\0{case['case_sha256']}".encode()
    while len(values) < count:
        digest = hashlib.sha256(seed + index.to_bytes(8, "little")).digest()
        candidate = int.from_bytes(digest[:8], "little") % vocab_size
        index += 1
        if candidate in reserved:
            continue
        values.append(candidate)
    return values


def atomic_write(path: Path, value: Any) -> None:
    reject_symlink_components(path.parent)
    if path.exists() or path.is_symlink():
        raise FixtureError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.incomplete")
    with temporary.open("xb") as stream:
        stream.write((json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def generate(expanded: dict[str, Any], expanded_sha256: str, served: dict[str, Any], served_sha256: str, output_dir: Path, subset: str) -> dict[str, Any]:
    all_cases = validate_expanded(expanded, served)
    stages = expanded.get("stage_case_count")
    if not isinstance(stages, dict) or any(not isinstance(value, int) for value in stages.values()):
        raise FixtureError("expanded stage_case_count is missing")
    if subset not in SUBSETS:
        raise FixtureError(f"subset must be one of {', '.join(SUBSETS)}")
    selected = all_cases if subset == "all" else [case for case in all_cases if case.get("stage_id") == subset]
    if not selected:
        raise FixtureError(f"subset has no cases: {subset}")
    expected = len(all_cases) if subset == "all" else stages.get(subset)
    if expected != len(selected):
        raise FixtureError(f"subset case count differs: expected {expected}, got {len(selected)}")
    reserved = set(served["reserved_token_ids"])
    # The served-model validation includes reasoning ids in this digest, but intentionally does not
    # expose the reserved-id list in the public index.
    fixture_root = output_dir / "cases"
    entries: list[dict[str, Any]] = []
    for case in sorted(selected, key=lambda item: item["case_id"]):
        count = case["prompt_tokens"]
        ids = token_ids(case, count, served["vocab_size"], reserved)
        fixture = {"schema_version": FIXTURE_SCHEMA, "cases": [{"case_id": case["case_id"], "prompt_token_ids": ids, "step_count": case["generated_tokens"]}]}
        fixture_path = fixture_root / f"{case['case_id']}.json"
        atomic_write(fixture_path, fixture)
        fixture_bytes = canonical(fixture)
        # The driver reads the regular JSON representation; hash that exact representation.
        fixture_hash = file_sha256(fixture_path, "fixture")
        entries.append({
            "case_id": case["case_id"],
            "case_sha256": case["case_sha256"],
            "fixture_path": str(fixture_path),
            "fixture_sha256": fixture_hash,
            "prompt_tokens": count,
            "context_tokens": case["context_tokens"],
            "generated_tokens": case["generated_tokens"],
            "prompt_token_ids_sha256": sha256_bytes(canonical(ids)),
            "driver_compatibility": "pure_prefill" if case["scope"] in SUPPORTED_PREFILL_SCOPES and case["phase"] == "cold_prefill" else "unsupported_scope_or_phase",
            "runnable_reason": None if case["scope"] in SUPPORTED_PREFILL_SCOPES and case["phase"] == "cold_prefill" else "driver supports full_model cold_prefill only",
        })
    index = {
        "schema_version": INDEX_SCHEMA,
        "expanded_manifest_sha256": expanded_sha256,
        "served_model_manifest_sha256": served_sha256,
        "subset": subset,
        "case_count": len(entries),
        "contract": {
            "context_length": served["context_length"],
            "vocab_size": served["vocab_size"],
            "max_completion_tokens": served["max_completion_tokens"],
            "eos_token_ids_sha256": sha256_bytes(canonical(served["eos_token_ids"])),
            "reserved_token_ids_sha256": served["reserved_token_ids_sha256"],
            "tokenizer_files_sha256": served["tokenizer_files_sha256"],
        },
        "cases": entries,
    }
    return index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expanded", type=Path, required=True)
    parser.add_argument("--served-model-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-index", type=Path, required=True)
    parser.add_argument("--subset", choices=SUBSETS, default="all")
    args = parser.parse_args(argv)
    try:
        expanded, expanded_raw = read_regular(args.expanded, "expanded")
        served, served_raw = read_regular(args.served_model_manifest, "served model")
        served_contract = validate_served_model(served, args.served_model_manifest.parent)
        index = generate(expanded, sha256_bytes(expanded_raw), served_contract, sha256_bytes(served_raw), args.output_dir, args.subset)
        atomic_write(args.output_index, index)
        print(json.dumps({"status": "ok", "subset": args.subset, "case_count": index["case_count"]}, sort_keys=True))
        return 0
    except (FixtureError, OSError, ValueError) as error:
        print(f"AQ4 P2 fixture generation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
