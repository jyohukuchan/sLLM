#!/usr/bin/env python3
"""Validate the Phase 6 A2 drift, reuse, reader, and dependency contract."""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ContractError, ROOT, read_json  # noqa: E402
from validate_rust_dependencies import validate_schema  # noqa: E402

CONTRACT_PATH = Path("ci/contracts/phase6-a2-v1.json")
SCHEMA_PATH = Path("ci/schema/phase6-a2-v1.schema.json")
POLICY_PATH = Path("ci/dependencies/rust-workspace-v1.json")
NORMATIVE_PIN = "117ce5680e4269f6656a4fd70d28f9755630d938"
CURRENT_OBSERVATION = "11854aef674352d3f9cd5c0a7038f079a7bbac06"
LLAMA_COMMIT = "f5919bf458ef190468b5c329bb293f8a54a1e69c"
EXPECTED_ENGINES = {
    "vLLM": "568afb3a13806beb53bb2e6bd518269357b237c0",
    "SGLang": "fdebc938f7f4d16fe6b9f55dcd9a767cf0899ea1",
    "TensorRT-LLM": "376f7e1bd8ed543f75014309e3fd4b237e9b0e73",
    "LMDeploy": "f4b8140ba19cd823c541241cbb113cc32f854e6a",
}
REQUIRED_DOC_MARKERS = {
    Path("docs/references/openai-chat-completions-v1-drift.md"): [NORMATIVE_PIN, CURRENT_OBSERVATION, "normative pin"],
    Path("docs/provenance/phase6-a2-llama-import-plan.md"): [LLAMA_COMMIT, "THIRD_PARTY_NOTICES.md", "facts-only"],
    Path("docs/references/phase6-openai-serving-reader.md"): [*EXPECTED_ENGINES, "facts-only", "acceptance cases"],
}


def _require_sha(value: Any, label: str, length: int) -> None:
    if not isinstance(value, str) or re.fullmatch(rf"[0-9a-f]{{{length}}}", value) is None:
        raise ContractError(f"{label} is not a lowercase {length}-hex digest")


def validate_contract(contract: dict[str, Any], policy: dict[str, Any], *, repo: Path = ROOT) -> None:
    drift = contract.get("openai_drift", {})
    normative = drift.get("normative", {})
    current = drift.get("current_observation", {})
    if normative.get("openapi_commit") != NORMATIVE_PIN or current.get("openapi_commit") != CURRENT_OBSERVATION:
        raise ContractError("OpenAI normative/current identity drifted")
    _require_sha(normative.get("openapi_sha256"), "normative OpenAPI SHA-256", 64)
    _require_sha(current.get("openapi_sha256"), "current OpenAPI SHA-256", 64)
    if drift.get("normative_pin_changed") is not False:
        raise ContractError("current observation must not silently change the normative pin")
    for field in ("request_changes", "response_changes", "stream_changes", "error_changes"):
        if drift.get(field) != []:
            raise ContractError(f"unexpected current OpenAPI {field}")
    if drift.get("other_changes") != [{"schema": "ModelIdsShared", "kind": "enum_added", "values": ["gpt-5.5"]}]:
        raise ContractError("current OpenAPI non-wire drift record changed")
    if not drift.get("supported_subset") or not drift.get("rejected_subset"):
        raise ContractError("supported and rejected profile subsets must remain explicit")

    reuse = contract.get("llama_reuse", {})
    if reuse.get("commit") != LLAMA_COMMIT or reuse.get("license") != "MIT" or reuse.get("actual_import_commit") is not None:
        raise ContractError("llama.cpp import-plan identity/state drifted")
    _require_sha(reuse.get("license_sha256"), "llama.cpp license SHA-256", 64)
    units = reuse.get("units")
    if not isinstance(units, list) or len(units) != 7:
        raise ContractError("llama.cpp reuse unit set drifted")
    paths: set[str] = set()
    for unit in units:
        if not isinstance(unit, dict) or set(unit) != {"path", "blob", "sha256", "reuse_mode", "scope", "planned_local"}:
            raise ContractError("llama.cpp reuse unit is not closed")
        if unit["path"] in paths or unit["reuse_mode"] not in {"ported", "adapted", "facts-only"}:
            raise ContractError("llama.cpp reuse path/mode is invalid")
        paths.add(unit["path"])
        _require_sha(unit["blob"], f"{unit['path']} blob", 40)
        _require_sha(unit["sha256"], f"{unit['path']} SHA-256", 64)
        if unit["reuse_mode"] == "facts-only" and unit["planned_local"] is not None:
            raise ContractError("facts-only unit must not have a planned local import")
    if "THIRD_PARTY_NOTICES.md" not in reuse.get("release_actions", []):
        raise ContractError("release provenance actions are incomplete")

    readers = contract.get("facts_only_readers")
    if not isinstance(readers, list) or {row.get("engine"): row.get("commit") for row in readers if isinstance(row, dict)} != EXPECTED_ENGINES:
        raise ContractError("facts-only engine revisions drifted")
    if any(not row.get("paths") for row in readers):
        raise ContractError("facts-only engine path set is empty")
    if len(contract.get("implementation_acceptance_cases", [])) != 8:
        raise ContractError("implementation acceptance case set drifted")

    dependency = contract.get("dependency_policy", {})
    counts = policy.get("counts", {})
    expected_counts = {key: counts.get(key) for key in ("packages", "registry_packages", "workspace_packages", "edges")}
    if dependency.get("path") != POLICY_PATH.as_posix() or dependency.get("counts") != expected_counts:
        raise ContractError("A2 dependency closure summary differs from the Rust policy")
    if dependency.get("workspace_package") != policy.get("feature_assertions", {}).get("server_runtime", {}).get("workspace_package"):
        raise ContractError("A2 server package differs from the Rust feature policy")

    for path, markers in REQUIRED_DOC_MARKERS.items():
        try:
            text = (repo / path).read_text(encoding="utf-8")
        except OSError as exc:
            raise ContractError(f"required A2 document is missing: {path}") from exc
        missing = [marker for marker in markers if marker not in text]
        if missing:
            raise ContractError(f"required A2 document markers missing from {path}: {missing}")


def check_llama_reference(contract: dict[str, Any], repo: Path = ROOT) -> None:
    checkout = repo / "reference/llama.cpp"
    process = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True, capture_output=True, check=False
    )
    if process.returncode != 0 or process.stdout.strip() != LLAMA_COMMIT:
        raise ContractError("local llama.cpp checkout is absent or not at the recorded commit")
    for unit in contract["llama_reuse"]["units"]:
        path = checkout / unit["path"]
        try:
            observed = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            raise ContractError(f"recorded llama.cpp source is missing: {unit['path']}") from exc
        if observed != unit["sha256"]:
            raise ContractError(f"recorded llama.cpp source SHA-256 drifted: {unit['path']}")


def validate(repo: Path = ROOT, *, check_reference: bool = False) -> None:
    contract = read_json(repo / CONTRACT_PATH)
    schema = read_json(repo / SCHEMA_PATH)
    policy = read_json(repo / POLICY_PATH)
    validate_schema(contract, schema, label="Phase 6 A2 contract")
    validate_contract(contract, policy, repo=repo)
    if check_reference:
        check_llama_reference(contract, repo)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-reference", action="store_true", help="also verify the optional local llama.cpp checkout")
    args = parser.parse_args(argv)
    try:
        validate(check_reference=args.check_reference)
    except (ContractError, OSError, ValueError) as exc:
        print(f"phase6 A2 contract: FAIL: {exc}", file=sys.stderr)
        return 1
    print("phase6 A2 contract: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
