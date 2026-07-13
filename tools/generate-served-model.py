#!/usr/bin/env python3
"""Materialize a served-model manifest from a deployment profile and live files."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import stat
import sys
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
LOADER_PATH = ROOT / "services/openai-gateway/src/ullm_openai_gateway/served_model.py"
PROFILE_SCHEMA = "ullm.served_model.profile.v1"
AQ4_EVIDENCE_SCHEMA = "ullm.aq4_resident_promotion_evidence.v1"


class GenerationError(RuntimeError):
    """Raised when a profile cannot be bound to immutable local files."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise GenerationError(f"failed to read {label}") from error
    if not isinstance(value, dict):
        raise GenerationError(f"{label} must be a JSON object")
    return value


def _receipt_value(receipt: dict[str, Any], path: Any, label: str) -> str:
    if (
        not isinstance(path, list)
        or not path
        or not all(isinstance(item, str) and item for item in path)
    ):
        raise GenerationError(f"{label} must be a nonempty string path")
    value: Any = receipt
    for component in path:
        if not isinstance(value, dict) or component not in value:
            raise GenerationError(f"{label} is absent from the promotion receipt")
        value = value[component]
    if not isinstance(value, str) or not value:
        raise GenerationError(f"{label} in the promotion receipt is invalid")
    return value


def _resolve_receipt_file(receipt_path: Path, raw_path: str, label: str) -> Path:
    relative = Path(raw_path)
    if relative.is_absolute() or not relative.parts or any(
        component in ("", ".", "..") for component in relative.parts
    ):
        raise GenerationError(f"{label} must be a safe relative path")
    unresolved = receipt_path.parent / relative
    if unresolved.is_symlink():
        raise GenerationError(f"{label} must identify a regular non-symlink file")
    resolved = unresolved.resolve()
    try:
        resolved.relative_to(receipt_path.parent.resolve())
    except ValueError as error:
        raise GenerationError(f"{label} escapes the promotion directory") from error
    if resolved.is_symlink() or not resolved.is_file():
        raise GenerationError(f"{label} must identify a regular non-symlink file")
    return resolved


def _validate_aq4_evidence(
    *,
    profile: dict[str, Any],
    promotion_profile: dict[str, Any],
    receipt: dict[str, Any],
    receipt_path: Path,
    source_commit: str,
    worker_binary: Path,
    worker_sha256: str,
    product_root: Path,
    package_manifest_path: str,
    package_manifest_sha256: str,
) -> None:
    required_schema = promotion_profile.get("required_schema_version")
    if required_schema is None:
        return
    if required_schema != "ullm.aq4_resident_promotion.v1":
        raise GenerationError("profile promotion receipt schema is unsupported")
    if receipt.get("schema_version") != required_schema:
        raise GenerationError("promotion receipt schema differs")

    evidence_path_value = _receipt_value(
        receipt,
        promotion_profile.get("evidence_from_receipt"),
        "AQ4 promotion evidence path",
    )
    evidence_sha256 = _receipt_value(
        receipt,
        promotion_profile.get("evidence_sha256_from_receipt"),
        "AQ4 promotion evidence SHA-256",
    )
    if len(evidence_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in evidence_sha256
    ):
        raise GenerationError("AQ4 promotion evidence SHA-256 is invalid")
    evidence_path = _resolve_receipt_file(
        receipt_path, evidence_path_value, "AQ4 promotion evidence path"
    )
    if _sha256_file(evidence_path) != evidence_sha256:
        raise GenerationError("AQ4 promotion evidence SHA-256 differs")
    evidence = _load_json(evidence_path, "AQ4 promotion evidence")
    if evidence.get("schema_version") != AQ4_EVIDENCE_SCHEMA:
        raise GenerationError("AQ4 promotion evidence schema differs")
    if evidence.get("verified") is not True:
        raise GenerationError("AQ4 promotion evidence is not verified")
    if evidence.get("production_receipt_written") is not False:
        raise GenerationError("AQ4 promotion evidence was not captured before receipt publication")
    gpu_preflight = evidence.get("gpu_exclusive_preflight")
    if not isinstance(gpu_preflight, dict) or set(gpu_preflight) != {
        "tool",
        "gpu_index",
        "positive_vram_processes",
    }:
        raise GenerationError("AQ4 promotion evidence GPU exclusivity preflight is missing")
    if (
        gpu_preflight.get("tool") != "rocm-smi --showpids --json"
        or gpu_preflight.get("gpu_index") != "1"
        or gpu_preflight.get("positive_vram_processes") != []
    ):
        raise GenerationError("AQ4 promotion evidence GPU exclusivity preflight failed")
    if evidence.get("source_commit") != source_commit:
        raise GenerationError("AQ4 promotion evidence source commit differs")
    if evidence.get("worker_binary") != os.fspath(worker_binary):
        raise GenerationError("AQ4 promotion evidence worker path differs")
    if evidence.get("worker_binary_sha256") != worker_sha256:
        raise GenerationError("AQ4 promotion evidence worker SHA-256 differs")

    _validate_aq4_token_comparisons(evidence)
    for mode in ("resident", "legacy"):
        result = evidence.get(mode)
        if not isinstance(result, dict) or result.get("clean_shutdown") is not True:
            raise GenerationError(f"AQ4 promotion evidence {mode} shutdown is not clean")
    child_checks = evidence["resident"].get("child_process_checks")
    if not isinstance(child_checks, list) or not child_checks or any(
        not isinstance(check, dict) or check.get("sibling_engine_count") != 0
        for check in child_checks
    ):
        raise GenerationError("AQ4 resident child-process evidence is incomplete")

    bundle = evidence.get("ephemeral_bundle")
    manifest = bundle.get("manifest") if isinstance(bundle, dict) else None
    if not isinstance(manifest, dict):
        raise GenerationError("AQ4 promotion evidence has no bound manifest")
    expected_worker = _required_object(profile, "worker")
    observed_worker = manifest.get("worker")
    if not isinstance(observed_worker, dict):
        raise GenerationError("AQ4 promotion evidence worker identity is absent")
    for name in ("protocol", "arguments", "required_environment", "identity"):
        if observed_worker.get(name) != expected_worker.get(name):
            raise GenerationError(f"AQ4 promotion evidence worker {name} differs")
    if observed_worker.get("binary") != os.fspath(worker_binary) or observed_worker.get(
        "binary_sha256"
    ) != worker_sha256:
        raise GenerationError("AQ4 promotion evidence worker binding differs")
    for name in ("public", "generation", "format"):
        if manifest.get(name) != _required_object(profile, name):
            raise GenerationError(f"AQ4 promotion evidence profile {name} differs")
    observed_product = manifest.get("product")
    observed_package = (
        observed_product.get("package") if isinstance(observed_product, dict) else None
    )
    if not isinstance(observed_package, dict) or observed_product.get("root") != os.fspath(
        product_root
    ):
        raise GenerationError("AQ4 promotion evidence product identity differs")
    if observed_package != {
        "manifest_path": package_manifest_path,
        "manifest_sha256": package_manifest_sha256,
    }:
        raise GenerationError("AQ4 promotion evidence package identity differs")
    if profile.get("worker", {}).get("protocol") == "ullm.worker.v2":
        _validate_v2_reasoning_evidence(evidence, manifest)


def _validate_aq4_token_comparisons(evidence: dict[str, Any]) -> None:
    comparisons = evidence.get("comparisons")
    resident = evidence.get("resident")
    legacy = evidence.get("legacy")
    resident_cases = resident.get("cases") if isinstance(resident, dict) else None
    legacy_cases = legacy.get("cases") if isinstance(legacy, dict) else None
    if (
        not isinstance(comparisons, list)
        or not comparisons
        or not isinstance(resident_cases, list)
        or not isinstance(legacy_cases, list)
    ):
        raise GenerationError("AQ4 promotion evidence comparisons are incomplete")

    def comparable_cases(cases: list[Any]) -> dict[str, list[int]]:
        result: dict[str, list[int]] = {}
        for case in cases:
            if not isinstance(case, dict) or case.get("id") == "reasoning-budget-zero":
                continue
            case_id = case.get("id")
            tokens = case.get("tokens")
            if (
                not isinstance(case_id, str)
                or not case_id
                or case_id in result
                or not isinstance(tokens, list)
                or not all(isinstance(token, int) and token >= 0 for token in tokens)
            ):
                raise GenerationError("AQ4 promotion evidence token cases are invalid")
            result[case_id] = tokens
        return result

    resident_by_id = comparable_cases(resident_cases)
    legacy_by_id = comparable_cases(legacy_cases)
    if resident_by_id.keys() != legacy_by_id.keys():
        raise GenerationError("AQ4 promotion evidence comparable case IDs differ")
    comparison_ids: set[str] = set()
    for item in comparisons:
        if not isinstance(item, dict):
            raise GenerationError("AQ4 promotion evidence comparisons are incomplete")
        case_id = item.get("id")
        if (
            not isinstance(case_id, str)
            or case_id in comparison_ids
            or item.get("tokens_exact_match") is not True
            or case_id not in resident_by_id
            or resident_by_id[case_id] != legacy_by_id[case_id]
        ):
            raise GenerationError("AQ4 promotion evidence token comparisons differ")
        comparison_ids.add(case_id)
    if comparison_ids != resident_by_id.keys():
        raise GenerationError("AQ4 promotion evidence comparisons are incomplete")


def _validate_v2_reasoning_evidence(
    evidence: dict[str, Any], manifest: dict[str, Any]
) -> None:
    """Recompute the deterministic v2 promotion case from raw token records."""

    reasoning = manifest.get("reasoning")
    worker = manifest.get("worker")
    if not isinstance(reasoning, dict) or not isinstance(worker, dict):
        raise GenerationError("AQ4 v2 promotion evidence lacks reasoning binding")
    resident = evidence.get("resident")
    legacy = evidence.get("legacy")
    if not isinstance(resident, dict) or not isinstance(legacy, dict):
        raise GenerationError("AQ4 v2 promotion evidence lacks worker results")
    resident_ready = resident.get("ready")
    legacy_ready = legacy.get("ready")
    if (
        not isinstance(resident_ready, dict)
        or not isinstance(legacy_ready, dict)
        or resident_ready.get("schema_version") != "ullm.worker.v2"
    ):
        raise GenerationError("AQ4 v2 resident ready schema differs")
    if legacy_ready.get("schema_version") != "ullm.worker.v1":
        raise GenerationError("AQ4 v2 legacy ready schema differs")

    resident_cases = resident.get("cases")
    legacy_cases = legacy.get("cases")
    if not isinstance(resident_cases, list) or not isinstance(legacy_cases, list):
        raise GenerationError("AQ4 v2 promotion evidence cases are incomplete")
    reasoning_cases = [
        case
        for case in resident_cases
        if isinstance(case, dict) and case.get("id") == "reasoning-budget-zero"
    ]
    if len(reasoning_cases) != 1:
        raise GenerationError("AQ4 v2 promotion evidence reasoning case is missing")
    raw_cases = [
        case
        for case in resident_cases
        if isinstance(case, dict) and case.get("id") != "reasoning-budget-zero"
    ]
    if len(raw_cases) != len(legacy_cases):
        raise GenerationError("AQ4 v2 promotion evidence raw case counts differ")
    reasoning_case = reasoning_cases[0]
    request = reasoning_case.get("reasoning")
    if not isinstance(request, dict):
        raise GenerationError("AQ4 v2 promotion reasoning request is absent")
    if (
        request.get("enabled") is not True
        or request.get("budget_tokens") != 0
        or request.get("dialect_id") != reasoning.get("dialect_id")
        or request.get("end_token_ids") != reasoning.get("end_token_ids")
        or request.get("forced_end_token_ids") != reasoning.get("forced_end_token_ids")
        or request.get("reserved_answer_tokens") != reasoning.get("reserved_answer_tokens")
    ):
        raise GenerationError("AQ4 v2 promotion reasoning request differs")
    usage = reasoning_case.get("reasoning_usage")
    forced_end = reasoning.get("forced_end_token_ids")
    reserved_answer = reasoning.get("reserved_answer_tokens")
    tokens = reasoning_case.get("tokens")
    if (
        not isinstance(usage, dict)
        or type(usage.get("reasoning_tokens")) is not int
        or usage.get("reasoning_tokens") != 0
        or not isinstance(forced_end, list)
        or type(reserved_answer) is not int
        or reserved_answer < 1
        or type(usage.get("forced_end_tokens")) is not int
        or usage.get("forced_end_tokens") != len(forced_end)
        or not isinstance(tokens, list)
        or not all(type(token) is int and token >= 0 for token in tokens)
        or len(tokens) < len(forced_end) + reserved_answer
        or tokens[: len(forced_end)] != forced_end
    ):
        raise GenerationError("AQ4 v2 promotion reasoning accounting is incomplete")


def _load_validator() -> ModuleType:
    package_root = LOADER_PATH.parents[1]
    if os.fspath(package_root) not in sys.path:
        sys.path.insert(0, os.fspath(package_root))
    package_name = "ullm_openai_gateway"
    if package_name not in sys.modules:
        package = ModuleType(package_name)
        package.__path__ = [os.fspath(package_root / package_name)]  # type: ignore[attr-defined]
        package.__package__ = package_name
        sys.modules[package_name] = package
    module_name = "ullm_openai_gateway.served_model"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, LOADER_PATH)
    if spec is None or spec.loader is None:
        raise GenerationError("served-model validator is unavailable")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def _required_object(parent: dict[str, Any], name: str) -> dict[str, Any]:
    value = parent.get(name)
    if not isinstance(value, dict):
        raise GenerationError(f"profile.{name} must be an object")
    return value


def materialize(profile_path: Path) -> dict[str, Any]:
    profile = _load_json(profile_path, "served-model profile")
    if profile.get("schema_version") != PROFILE_SCHEMA:
        raise GenerationError("served-model profile schema is unsupported")

    tokenizer_profile = _required_object(profile, "tokenizer")
    worker_profile = _required_object(profile, "worker")
    product_profile = _required_object(profile, "product")
    promotion_profile = _required_object(profile, "promotion")
    reasoning_profile = profile.get("reasoning")
    if reasoning_profile is not None:
        if not isinstance(reasoning_profile, dict):
            raise GenerationError("profile.reasoning must be an object")
        if worker_profile.get("protocol") != "ullm.worker.v2":
            raise GenerationError("profile.reasoning requires ullm.worker.v2")
    elif worker_profile.get("protocol") == "ullm.worker.v2":
        raise GenerationError("ullm.worker.v2 profile requires reasoning")

    tokenizer_root = Path(str(tokenizer_profile.get("root", ""))).resolve()
    tokenizer_config = _load_json(
        tokenizer_root / "tokenizer_config.json", "tokenizer config"
    )
    chat_template = tokenizer_config.get("chat_template")
    if not isinstance(chat_template, str) or not chat_template:
        raise GenerationError("tokenizer config has no string chat template")
    raw_tokenizer_files = tokenizer_profile.get("files")
    if not isinstance(raw_tokenizer_files, list) or not raw_tokenizer_files:
        raise GenerationError("profile.tokenizer.files must be a nonempty array")
    tokenizer_files: dict[str, str] = {}
    for item in raw_tokenizer_files:
        if not isinstance(item, str) or not item or item in tokenizer_files:
            raise GenerationError("profile.tokenizer.files is invalid")
        tokenizer_files[item] = _sha256_file(tokenizer_root / item)

    worker_binary = Path(str(worker_profile.get("binary", ""))).resolve()
    worker_sha256 = _sha256_file(worker_binary)
    product_root = Path(str(product_profile.get("root", ""))).resolve()
    package = _required_object(product_profile, "package")
    package_manifest_path = str(package.get("manifest_path", ""))
    package_manifest_sha256 = _sha256_file(product_root / package_manifest_path)

    receipt_path = Path(str(promotion_profile.get("receipt", ""))).resolve()
    receipt = _load_json(receipt_path, "promotion receipt")
    source_commit = _receipt_value(
        receipt,
        promotion_profile.get("source_commit_from_receipt"),
        "promotion source commit",
    )
    _validate_aq4_evidence(
        profile=profile,
        promotion_profile=promotion_profile,
        receipt=receipt,
        receipt_path=receipt_path,
        source_commit=source_commit,
        worker_binary=worker_binary,
        worker_sha256=worker_sha256,
        product_root=product_root,
        package_manifest_path=package_manifest_path,
        package_manifest_sha256=package_manifest_sha256,
    )

    artifact_profile = product_profile.get("artifact")
    artifact: dict[str, str] | None
    if artifact_profile is None:
        artifact = None
    elif isinstance(artifact_profile, dict):
        artifact_manifest_path = str(artifact_profile.get("manifest_path", ""))
        artifact = {
            "manifest_path": artifact_manifest_path,
            "manifest_sha256": _sha256_file(product_root / artifact_manifest_path),
            "content_sha256": _receipt_value(
                receipt,
                artifact_profile.get("content_sha256_from_receipt"),
                "artifact content SHA-256",
            ),
        }
    else:
        raise GenerationError("profile.product.artifact must be an object or null")

    document = {
        "schema_version": (
            "ullm.served_model.v2"
            if reasoning_profile is not None
            else "ullm.served_model.v1"
        ),
        "public": _required_object(profile, "public"),
        "generation": _required_object(profile, "generation"),
        "format": _required_object(profile, "format"),
        "tokenizer": {
            "root": os.fspath(tokenizer_root),
            "transformers_version": tokenizer_profile.get("transformers_version"),
            "class": tokenizer_profile.get("class"),
            "chat_template_sha256": hashlib.sha256(
                chat_template.encode("utf-8")
            ).hexdigest(),
            "files": tokenizer_files,
            "template_options": tokenizer_profile.get("template_options"),
        },
        "worker": {
            "protocol": worker_profile.get("protocol"),
            "binary": os.fspath(worker_binary),
            "binary_sha256": worker_sha256,
            "arguments": worker_profile.get("arguments"),
            "required_environment": worker_profile.get("required_environment"),
            "identity": worker_profile.get("identity"),
        },
        "product": {
            "root": os.fspath(product_root),
            "artifact": artifact,
            "package": {
                "manifest_path": package_manifest_path,
                "manifest_sha256": package_manifest_sha256,
            },
        },
        "promotion": {
            "source_commit": source_commit,
            "receipt": os.fspath(receipt_path),
            "receipt_sha256": _sha256_file(receipt_path),
        },
    }
    if reasoning_profile is not None:
        document["reasoning"] = reasoning_profile
    return document


def generate(profile_path: Path, output_path: Path) -> str:
    document = materialize(profile_path)
    if output_path.is_symlink():
        raise GenerationError("output path must not be a symlink")
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(document, ensure_ascii=True, allow_nan=False, indent=2) + "\n"
    ).encode("utf-8")
    temporary: Path | None = None
    try:
        descriptor, raw_path = tempfile.mkstemp(
            prefix=f".{output_path.name}.", dir=output_path.parent
        )
        temporary = Path(raw_path)
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(encoded)
            destination.flush()
            os.fsync(destination.fileno())
        temporary.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
        model = _load_validator().load_served_model(temporary)
        os.replace(temporary, output_path)
        temporary = None
        directory = os.open(output_path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        return model.manifest_sha256
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        digest = generate(args.profile, args.output)
    except Exception as error:
        print(f"served-model generation failed: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "schema_version": "ullm.served_model.generation.v1",
                "manifest_sha256": digest,
                "output": os.fspath(args.output.resolve()),
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
