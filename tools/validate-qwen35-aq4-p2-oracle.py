#!/usr/bin/env python3
"""Independently validate Qwen3.5-9B AQ4 P2 source/path oracle evidence."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qwen35_aq4_p2_oracle as oracle  # noqa: E402


PATH_RUNTIME_SCHEMA = "ullm.qwen35_aq4_path_oracle_runtime.v1"
PRODUCTION_DEVICE_INDEX = 1
PRODUCTION_VISIBLE_DEVICES = "1"
PRODUCTION_DEVICE_ARCHITECTURE = "gfx1201"
PRODUCTION_EXECUTION_PROFILE = "rdna4_aq4_resident"
REQUIRED_HIP_KERNEL_ENV = (
    "ULLM_REQUIRE_HIP_AQ4_KERNEL",
    "ULLM_REQUIRE_HIP_AQ4_MATVEC_KERNEL",
    "ULLM_REQUIRE_HIP_AQ4_MATVEC_BATCH_KERNEL",
    "ULLM_REQUIRE_HIP_AQ4_REGISTER_BM8_KERNEL",
    "ULLM_REQUIRE_HIP_AQ4_MATVEC_ADD_KERNEL",
    "ULLM_REQUIRE_HIP_AQ4_MATVEC_PAIR_KERNEL",
    "ULLM_REQUIRE_HIP_AQ4_MATVEC_TRIPLE_KERNEL",
    "ULLM_REQUIRE_HIP_AQ4_MATVEC_QKV_Z_GATE_BETA_KERNEL",
    "ULLM_REQUIRE_HIP_ADD_KERNEL",
    "ULLM_REQUIRE_HIP_BF16_MATVEC_KERNEL",
    "ULLM_REQUIRE_HIP_BF16_ROW_KERNEL",
    "ULLM_REQUIRE_HIP_LINEAR_ATTN_GATE_BETA_KERNEL",
    "ULLM_REQUIRE_HIP_LINEAR_ATTN_KERNEL",
    "ULLM_REQUIRE_HIP_LINEAR_ATTN_QKV_PREPARE_BATCH_KERNEL",
    "ULLM_REQUIRE_HIP_LINEAR_ATTN_RECURRENT_KERNEL",
    "ULLM_REQUIRE_HIP_LINEAR_ATTN_RECURRENT_SEQUENCE_KERNEL",
    "ULLM_REQUIRE_HIP_PAGED_KV_WRITE_CHUNK_KERNEL",
    "ULLM_REQUIRE_HIP_PAGED_CAUSAL_GQA_CHUNK_KERNEL",
    "ULLM_REQUIRE_HIP_PAGED_DECODE_ATTN_KERNEL",
    "ULLM_REQUIRE_HIP_PAGED_DECODE_SPLIT_KERNEL",
    "ULLM_REQUIRE_HIP_PAGED_KV_WRITE_KERNEL",
    "ULLM_REQUIRE_HIP_QWEN35_Q_SPLIT_KERNEL",
    "ULLM_REQUIRE_HIP_QWEN35_QK_NORM_ROPE_BATCH_KERNEL",
    "ULLM_REQUIRE_HIP_QWEN35_QK_NORM_ROPE_PAGED_KV_WRITE_KERNEL",
    "ULLM_REQUIRE_HIP_RMSNORM_KERNEL",
    "ULLM_REQUIRE_HIP_ROPE_KERNEL",
    "ULLM_REQUIRE_HIP_SEGMENTED_RMSNORM_SILU_MUL_KERNEL",
    "ULLM_REQUIRE_HIP_SIGMOID_MUL_KERNEL",
    "ULLM_REQUIRE_HIP_SILU_MUL_KERNEL",
    "ULLM_REQUIRE_HIP_TOP1_KERNEL",
)
VALIDATION_TEST_HOOK = None


def _validation_hook(stage: str, root: Path) -> None:
    hook = VALIDATION_TEST_HOOK
    if hook is not None:
        hook(stage, root)


def _exact(value: Any, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        actual = set(value) if isinstance(value, dict) else set()
        raise oracle.OracleError(
            f"{label} keys differ: missing={sorted(fields - actual)} extra={sorted(actual - fields)}"
        )
    return value


def _absolute_regular(
    raw: Any,
    label: str,
    *,
    context: oracle.ValidationContext | None = None,
    maximum: int | None = None,
) -> Path:
    if not isinstance(raw, str) or not raw or not Path(raw).is_absolute():
        raise oracle.OracleError(f"{label} must be an absolute path")
    path = Path(raw)
    # sha256_file performs O_NOFOLLOW, single-link, fd/path identity validation.
    if maximum is not None:
        oracle.read_regular_bytes(path, label, maximum, context=context)
    else:
        _sha(path, context=context)
    return path.resolve(strict=True)


def _absolute_directory(
    raw: Any, label: str, *, context: oracle.ValidationContext | None = None
) -> Path:
    if not isinstance(raw, str) or not raw or not Path(raw).is_absolute():
        raise oracle.OracleError(f"{label} must be an absolute path")
    path = Path(raw)
    if path.is_symlink() or not path.is_dir():
        raise oracle.OracleError(f"{label} must be a non-symlink directory")
    resolved = path.resolve(strict=True)
    if context is not None:
        context.snapshot_directory(resolved, label)
    return resolved


def _relative_under(root: Path, raw: Any, label: str) -> Path:
    if not isinstance(raw, str):
        raise oracle.OracleError(f"{label} must be a relative path")
    path = oracle.safe_relative(root, raw, label)
    return path.resolve(strict=True)


def _replay_sha256(token_ids: list[int]) -> str:
    if not token_ids:
        raise oracle.OracleError("source replay token sequence is empty")
    digest = hashlib.sha256()
    digest.update(b"ullm.qwen35_aq4.calibration_replay.v1\0")
    digest.update(len(token_ids).to_bytes(8, "little", signed=False))
    for token_id in token_ids:
        digest.update(oracle.integer(token_id, "source replay token", minimum=0).to_bytes(8, "little", signed=False))
    return digest.hexdigest()


def _sha(path: Path, *, context: oracle.ValidationContext | None = None) -> str:
    return oracle.sha256_file(path, context=context)


def _rehash_files(root_raw: str, files: list[dict[str, Any]], label: str) -> None:
    root = Path(root_raw)
    if root.is_symlink() or not root.is_dir():
        raise oracle.OracleError(f"{label} root is missing, symlinked, or not a directory")
    for entry in files:
        path = oracle.safe_relative(root, entry["file"], f"{label} {entry['file']}")
        if path.stat().st_size != entry["bytes"] or _sha(path) != entry["sha256"]:
            raise oracle.OracleError(f"{label} file identity differs: {entry['file']}")


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _validate_sha256s(
    root: Path, manifest: dict[str, Any], context: oracle.ValidationContext
) -> None:
    sums_path = oracle.safe_relative(root, "SHA256SUMS", "SHA256SUMS")
    expected_order = ["manifest.json", manifest["payload"]["file"], "runtime.json"]
    expected_names = set(expected_order)
    entries: dict[str, str] = {}
    try:
        sums_text = oracle.read_regular_bytes(
            sums_path, "SHA256SUMS", 4096, context=context
        ).decode("ascii")
    except UnicodeError as error:
        raise oracle.OracleError("SHA256SUMS must be ASCII") from error
    if not sums_text.endswith("\n") or "\r" in sums_text:
        raise oracle.OracleError("SHA256SUMS framing differs")
    lines = sums_text[:-1].split("\n")
    for line_number, line in enumerate(lines, 1):
        parts = line.split("  ")
        if len(parts) != 2 or parts[1] in entries:
            raise oracle.OracleError(f"SHA256SUMS line {line_number} is invalid or duplicate")
        oracle.ensure_sha256(parts[0], f"SHA256SUMS line {line_number}")
        entries[parts[1]] = parts[0]
    if set(entries) != expected_names:
        raise oracle.OracleError("SHA256SUMS coverage differs")
    if list(entries) != expected_order:
        raise oracle.OracleError("SHA256SUMS order differs")
    actual_names = set()
    for path in root.iterdir():
        info = path.lstat()
        if path.is_symlink() or not path.is_file() or info.st_nlink != 1:
            raise oracle.OracleError("oracle root contains a non-regular artifact")
        actual_names.add(path.name)
    if actual_names != expected_names | {"SHA256SUMS"}:
        raise oracle.OracleError("oracle root file coverage differs from SHA256SUMS")
    for name, digest in entries.items():
        path = oracle.safe_relative(root, name, f"SHA256SUMS target {name}")
        if _sha(path, context=context) != digest:
            raise oracle.OracleError(f"SHA256SUMS digest differs: {name}")
    _validation_hook("after_sha256s", root)


def _package_binding_status(
    artifact: dict[str, Any], context: oracle.ValidationContext
) -> tuple[bool, str | None]:
    """Recognize the extended package-only binding before runtime reconstruction."""
    if artifact.get("artifact_binding_kind") != "package_manifest":
        return False, "path oracle artifact binding kind is not package_manifest"
    required = {"served_model_manifest_path", "served_model_manifest_sha256", "served_package_manifest_sha256"}
    if not required.issubset(artifact):
        return False, "path oracle package binding lacks active served-model identity"
    manifest_path = Path(artifact["served_model_manifest_path"])
    if manifest_path.is_symlink() or not manifest_path.is_file():
        return False, "active served-model manifest is not a regular file"
    active = oracle.load_json(manifest_path, context=context)
    if _sha(manifest_path, context=context) != artifact["served_model_manifest_sha256"]:
        return False, "active served-model manifest SHA-256 differs"
    try:
        product = active["product"]
        package = product["package"]
        active_artifact = product.get("artifact")
        active_package_sha = package["manifest_sha256"]
    except (KeyError, TypeError) as error:
        raise oracle.OracleError("active served-model package identity is invalid") from error
    if active_artifact is not None:
        return False, "active served-model product exposes an artifact; package-only binding is invalid"
    if active_package_sha != artifact["package_manifest_sha256"] or active_package_sha != artifact["served_package_manifest_sha256"]:
        return False, "active served-model package manifest SHA-256 differs"
    return True, None


def _load_cases_input(
    path: Path, context: oracle.ValidationContext
) -> tuple[list[dict[str, Any]], dict[str, list[int]]]:
    value = oracle.load_json(path, context=context)
    if not isinstance(value, dict) or set(value) != {"cases"} or not isinstance(value["cases"], list):
        raise oracle.OracleError("path runtime cases input schema differs")
    normalized: list[dict[str, Any]] = []
    prompts: dict[str, list[int]] = {}
    seen: set[str] = set()
    for index, raw in enumerate(value["cases"]):
        item = _exact(raw, {"case_id", "prompt_token_ids", "step_count"}, f"path runtime cases[{index}]")
        case_id = item["case_id"]
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise oracle.OracleError("path runtime case IDs differ")
        seen.add(case_id)
        token_ids = item["prompt_token_ids"]
        if not isinstance(token_ids, list) or not token_ids or len(token_ids) > 4096:
            raise oracle.OracleError("path runtime prompt token list differs")
        prompts[case_id] = [oracle.integer(token, "path runtime prompt token", minimum=0) for token in token_ids]
        step_count = oracle.integer(item["step_count"], "path runtime step count", minimum=1)
        if step_count > oracle.MAX_STEPS:
            raise oracle.OracleError("path runtime step count exceeds bound")
        normalized.append(
            {
                "case_id": case_id,
                "prompt_token_count": len(token_ids),
                "prompt_token_ids_sha256": oracle.canonical_token_ids_hash(token_ids),
                "step_count": step_count,
            }
        )
    if not normalized or len(normalized) > oracle.MAX_CASES:
        raise oracle.OracleError("path runtime cases exceed bound")
    return normalized, prompts


def _validate_source_replay(
    value: Any,
    manifest: dict[str, Any],
    *,
    production: bool,
    source_context: oracle.ValidationContext,
    binding_context: oracle.ValidationContext,
) -> None:
    replay = _exact(
        value,
        {"cases", "cases_input", "manifest_sha256", "payload_sha256", "root"},
        "path runtime source_replay",
    )
    source_root = _absolute_directory(replay["root"], "path runtime source root")
    source_manifest = oracle.validate_manifest(
        source_root, expected_kind="source", context=source_context
    )
    if production and source_manifest["evidence_class"] != "production":
        raise oracle.OracleError("production path runtime requires a production source oracle")
    if _sha(source_root / "manifest.json", context=source_context) != replay["manifest_sha256"]:
        raise oracle.OracleError("path runtime source manifest hash differs")
    if source_manifest["payload"]["sha256"] != replay["payload_sha256"]:
        raise oracle.OracleError("path runtime source payload hash differs")
    if source_manifest["cases"] != manifest["cases"]:
        raise oracle.OracleError("path runtime source/path case contract differs")
    cases_input = _exact(replay["cases_input"], {"path", "sha256"}, "path runtime cases input")
    cases_path = _absolute_regular(
        cases_input["path"],
        "path runtime cases input",
        context=binding_context,
        maximum=oracle.MAX_JSON_BYTES,
    )
    if _sha(cases_path, context=binding_context) != cases_input["sha256"]:
        raise oracle.OracleError("path runtime cases input hash differs")
    normalized_cases, prompts = _load_cases_input(cases_path, binding_context)
    if normalized_cases != manifest["cases"]:
        raise oracle.OracleError("path runtime cases input differs from path manifest")
    by_case = {case["case_id"]: [] for case in manifest["cases"]}
    for row in oracle.payload_records(source_root, source_manifest, context=source_context):
        if row["case_id"] not in by_case:
            raise oracle.OracleError("path runtime source replay contains an unknown case")
        by_case[row["case_id"]].append(row["greedy_token_id"])
    bindings = replay["cases"]
    if not isinstance(bindings, list) or len(bindings) != len(manifest["cases"]):
        raise oracle.OracleError("path runtime replay case coverage differs")
    expected_bindings = []
    for case in manifest["cases"]:
        case_id = case["case_id"]
        tokens = by_case[case_id]
        if len(tokens) != case["step_count"]:
            raise oracle.OracleError(f"path runtime replay length differs for {case_id}")
        prompt = prompts[case_id]
        expected_bindings.append(
            {
                "case_id": case_id,
                "length": len(tokens),
                "source_sequence_sha256": _replay_sha256(tokens),
                "contexts": [
                    {
                        "step": step,
                        "length": len(prompt) + step,
                        "token_ids_sha256": oracle.canonical_token_ids_hash(prompt + tokens[:step]),
                    }
                    for step in range(case["step_count"])
                ],
            }
        )
    if bindings != expected_bindings:
        raise oracle.OracleError("path runtime replay case/hash/context binding differs")


def _validate_served_guard(
    guard_value: Any,
    artifact: dict[str, Any],
    runtime: dict[str, Any],
    manifest: dict[str, Any],
    *,
    production: bool,
    context: oracle.ValidationContext,
) -> None:
    guard = _exact(
        guard_value,
        {
            "evidence_scope",
            "manifest",
            "manifest_sha256",
            "package",
            "public",
            "required_environment",
            "required_environment_sha256",
            "worker",
        },
        "path runtime served_model_guard",
    )
    if guard["evidence_scope"] != runtime["evidence_scope"]:
        raise oracle.OracleError("path runtime served guard evidence scope differs")
    required = list(REQUIRED_HIP_KERNEL_ENV) if guard["manifest"] is not None else []
    if guard["required_environment"] != required or guard["required_environment_sha256"] != oracle.canonical_sha256(required):
        raise oracle.OracleError("path runtime served guard environment differs")
    expected_environment = {name: "1" for name in required}
    if runtime["visible_devices"] is not None:
        expected_environment |= {
            "HIP_VISIBLE_DEVICES": runtime["visible_devices"],
            "ULLM_HIP_VISIBLE_DEVICES": runtime["visible_devices"],
        }
    if runtime["execution_environment"] != expected_environment:
        raise oracle.OracleError("path runtime execution environment differs")
    if not production:
        if guard["worker"] is not None or guard["package"] is not None or guard["public"] is not None:
            raise oracle.OracleError("fixture path runtime must not claim production served identity")
        if guard["manifest"] is None:
            if guard["manifest_sha256"] is not None:
                raise oracle.OracleError("fixture served manifest hash differs")
        else:
            served_path = _absolute_regular(
                guard["manifest"],
                "fixture served-model manifest",
                context=context,
                maximum=oracle.MAX_JSON_BYTES,
            )
            if _sha(served_path, context=context) != guard["manifest_sha256"]:
                raise oracle.OracleError("fixture served-model manifest hash differs")
        return

    if artifact.get("artifact_binding_kind") not in {"package_manifest", "artifact_manifest"}:
        raise oracle.OracleError("production path manifest lacks extended served binding")
    served_path = _absolute_regular(
        guard["manifest"],
        "production served-model manifest",
        context=context,
        maximum=oracle.MAX_JSON_BYTES,
    )
    served_sha = _sha(served_path, context=context)
    if served_sha != guard["manifest_sha256"] or served_sha != artifact.get("served_model_manifest_sha256"):
        raise oracle.OracleError("production served-model manifest hash chain differs")
    if served_path != Path(artifact.get("served_model_manifest_path", "")).resolve(strict=True):
        raise oracle.OracleError("production served-model manifest path chain differs")
    served = oracle.load_json(served_path, context=context)
    if not isinstance(served, dict) or served.get("schema_version") != "ullm.served_model.v2":
        raise oracle.OracleError("production served-model schema is not v2")
    try:
        public = served["public"]
        worker = served["worker"]
        worker_identity = worker["identity"]
        product = served["product"]
        package = product["package"]
        active_artifact = product.get("artifact")
    except (KeyError, TypeError) as error:
        raise oracle.OracleError("production served-model identity is incomplete") from error
    public_binding = _exact(guard["public"], {"model_id", "model_revision"}, "served public binding")
    if public_binding != {"model_id": public.get("upstream_id"), "model_revision": public.get("revision")} or public_binding["model_id"] != manifest["identity"]["model_id"]:
        raise oracle.OracleError("production served public model identity differs")
    worker_binding = _exact(
        guard["worker"],
        {"binary_path", "binary_sha256", "device_architecture", "execution_profile"},
        "served worker binding",
    )
    worker_path = _absolute_regular(
        worker_binding["binary_path"], "served worker binary", context=context
    )
    worker_sha = oracle.ensure_sha256(worker_binding["binary_sha256"], "served worker binary hash")
    if str(worker_path) != worker.get("binary") or worker_sha != worker.get("binary_sha256") or _sha(worker_path, context=context) != worker_sha:
        raise oracle.OracleError("served worker binary identity differs")
    if (
        worker_identity.get("device") != PRODUCTION_DEVICE_ARCHITECTURE
        or worker_identity.get("execution_profile") != PRODUCTION_EXECUTION_PROFILE
        or worker_binding["device_architecture"] != PRODUCTION_DEVICE_ARCHITECTURE
        or worker_binding["execution_profile"] != PRODUCTION_EXECUTION_PROFILE
    ):
        raise oracle.OracleError("served worker is not the bound R9700 execution profile")
    if worker.get("required_environment") != list(REQUIRED_HIP_KERNEL_ENV):
        raise oracle.OracleError("served worker required environment differs")
    package_binding = _exact(
        guard["package"],
        {"manifest_path", "manifest_sha256", "product_root"},
        "served package binding",
    )
    product_root = _absolute_directory(
        package_binding["product_root"], "served product root", context=context
    )
    declared_product_root = Path(product.get("root", ""))
    if not declared_product_root.is_absolute():
        declared_product_root = served_path.parent / declared_product_root
    if _absolute_directory(
        str(declared_product_root), "declared served product root", context=context
    ) != product_root:
        raise oracle.OracleError("served product root binding differs")
    package_path = _relative_under(product_root, package.get("manifest_path"), "served package manifest")
    package_sha = oracle.ensure_sha256(package.get("manifest_sha256"), "served package manifest hash")
    if (
        package_path != Path(package_binding["manifest_path"]).resolve(strict=True)
        or package_path != Path(runtime["package_manifest"]).resolve(strict=True)
        or _sha(package_path, context=context) != package_sha
        or package_binding["manifest_sha256"] != package_sha
        or runtime["package_manifest_sha256"] != package_sha
        or artifact["package_manifest_sha256"] != package_sha
        or artifact.get("served_package_manifest_sha256") != package_sha
    ):
        raise oracle.OracleError("served package manifest hash/path chain differs")
    binding_kind = artifact["artifact_binding_kind"]
    if binding_kind == "package_manifest":
        if active_artifact is not None or runtime["artifact_manifest"] is not None or runtime["artifact_manifest_sha256"] is not None:
            raise oracle.OracleError("package-only served binding exposes an artifact manifest")
    else:
        active = _exact(active_artifact, {"content_sha256", "manifest_path", "manifest_sha256"}, "served artifact")
        artifact_path = _relative_under(product_root, active["manifest_path"], "served artifact manifest")
        artifact_sha = oracle.ensure_sha256(active["manifest_sha256"], "served artifact manifest hash")
        if (
            artifact_path != Path(runtime["artifact_manifest"]).resolve(strict=True)
            or _sha(artifact_path, context=context) != artifact_sha
            or runtime["artifact_manifest_sha256"] != artifact_sha
            or artifact["artifact_manifest_sha256"] != artifact_sha
        ):
            raise oracle.OracleError("served artifact manifest hash/path chain differs")


def _validate_path_runtime(
    root: Path,
    manifest: dict[str, Any],
    context: oracle.ValidationContext,
    source_context: oracle.ValidationContext,
) -> dict[str, Any]:
    runtime_path = oracle.safe_relative(root, "runtime.json", "path runtime")
    runtime = oracle.load_json(runtime_path, context=context)
    runtime = _exact(
        runtime,
        {
            "all_m1",
            "artifact_manifest",
            "artifact_manifest_sha256",
            "binary",
            "device_index",
            "device_kind",
            "dtype",
            "evidence_scope",
            "execution_environment",
            "model_loads",
            "package_dir",
            "package_manifest",
            "package_manifest_sha256",
            "run",
            "runtime",
            "schema_version",
            "served_model_guard",
            "source_replay",
            "visible_devices",
        },
        "path runtime",
    )
    production = manifest["evidence_class"] == "production"
    expected_scope = "production_gpu" if production else "fixture_only"
    if runtime["schema_version"] != PATH_RUNTIME_SCHEMA or runtime["runtime"] != "ullm-aq4-p2-path-oracle":
        raise oracle.OracleError("path runtime schema/implementation differs")
    if runtime["evidence_scope"] != expected_scope:
        raise oracle.OracleError("path runtime evidence scope differs")
    if runtime["device_kind"] not in {"cpu", "gpu"} or runtime["dtype"] != "f32" or runtime["all_m1"] is not True or runtime["model_loads"] != 1:
        raise oracle.OracleError("path runtime execution contract differs")
    device_index = oracle.integer(runtime["device_index"], "path runtime device index", minimum=0)
    visible = runtime["visible_devices"]
    if visible is not None and (not isinstance(visible, str) or not visible.isdecimal() or str(int(visible)) != visible):
        raise oracle.OracleError("path runtime visible device mapping differs")
    if production:
        if runtime["device_kind"] != "gpu" or device_index != PRODUCTION_DEVICE_INDEX or visible != PRODUCTION_VISIBLE_DEVICES:
            raise oracle.OracleError("production path runtime is not bound to the R9700 GPU mapping")
    elif runtime["device_kind"] == "cpu" and (device_index != 0 or visible is not None):
        raise oracle.OracleError("CPU path fixture device mapping differs")
    package_dir = _absolute_directory(
        runtime["package_dir"], "path runtime package directory", context=context
    )
    package_path = _absolute_regular(
        runtime["package_manifest"], "path runtime package manifest", context=context
    )
    try:
        package_path.relative_to(package_dir)
    except ValueError as error:
        raise oracle.OracleError("path runtime package manifest escapes package directory") from error
    package_sha = oracle.ensure_sha256(runtime["package_manifest_sha256"], "path runtime package hash")
    artifact = manifest["identity"]["artifact"]
    if _sha(package_path, context=context) != package_sha or package_sha != artifact["package_manifest_sha256"]:
        raise oracle.OracleError("path runtime package manifest identity differs")
    if runtime["artifact_manifest"] is None:
        if runtime["artifact_manifest_sha256"] is not None or artifact["artifact_manifest_sha256"] is not None:
            raise oracle.OracleError("path runtime artifact nullability differs")
    else:
        artifact_path = _absolute_regular(
            runtime["artifact_manifest"], "path runtime artifact manifest", context=context
        )
        artifact_sha = oracle.ensure_sha256(runtime["artifact_manifest_sha256"], "path runtime artifact hash")
        if artifact_path == package_path or _sha(artifact_path, context=context) != artifact_sha or artifact_sha != artifact["artifact_manifest_sha256"]:
            raise oracle.OracleError("path runtime artifact manifest identity differs")
    binary = _exact(runtime["binary"], {"path", "sha256"}, "path runtime binary")
    binary_path = _absolute_regular(binary["path"], "path runtime binary", context=context)
    if _sha(binary_path, context=context) != oracle.ensure_sha256(binary["sha256"], "path runtime binary hash"):
        raise oracle.OracleError("path runtime binary hash differs")
    run = _exact(runtime["run"], {"elapsed_seconds", "row_count"}, "path runtime run")
    elapsed = oracle.finite(run["elapsed_seconds"], "path runtime elapsed_seconds")
    if elapsed <= 0 or run["row_count"] != manifest["payload"]["record_count"]:
        raise oracle.OracleError("path runtime run summary differs")
    _validate_source_replay(
        runtime["source_replay"],
        manifest,
        production=production,
        source_context=source_context,
        binding_context=context,
    )
    _validate_served_guard(
        runtime["served_model_guard"],
        artifact,
        runtime,
        manifest,
        production=production,
        context=context,
    )
    _validation_hook("after_path_semantics", root)
    _validate_sha256s(root, manifest, context)
    return runtime


def _validate_runtime(
    root: Path,
    manifest: dict[str, Any],
    context: oracle.ValidationContext | None = None,
) -> None:
    owned_context = context is None
    context = context or oracle.ValidationContext()
    runtime_path = oracle.safe_relative(root, "runtime.json", "runtime.json")
    runtime = oracle.load_json(runtime_path, context=context)
    if runtime != manifest.get("runtime"):
        raise oracle.OracleError("manifest runtime and runtime.json differ")
    expected_keys = {"device", "dtype", "full_vocab_ranking", "inference_mode", "low_cpu_mem_usage", "low_cpu_mem_usage_blocker", "max_resident_logit_rows", "model_loads", "preflight", "python", "run", "runtime", "safetensors", "torch", "torch_num_interop_threads", "torch_num_threads", "transformers"}
    if not isinstance(runtime, dict) or set(runtime) != expected_keys:
        raise oracle.OracleError("runtime keys differ")
    if runtime["runtime"] != "transformers.AutoModelForCausalLM" or runtime["device"] != "cpu" or runtime["dtype"] != "bfloat16":
        raise oracle.OracleError("runtime CPU/BF16 identity differs")
    if runtime["low_cpu_mem_usage"] is not False or runtime["low_cpu_mem_usage_blocker"] != "accelerate package is unavailable in the installed environment":
        raise oracle.OracleError("runtime low-memory loader status differs")
    if runtime["inference_mode"] is not True or runtime["full_vocab_ranking"] is not True or runtime["max_resident_logit_rows"] != 1 or runtime["model_loads"] != 1:
        raise oracle.OracleError("runtime bounded forward contract differs")
    if runtime["torch_num_threads"] != 1 or runtime["torch_num_interop_threads"] != 1:
        raise oracle.OracleError("runtime thread count differs")
    if runtime["python"] != platform.python_version() or runtime["torch"] != _package_version("torch") or runtime["transformers"] != _package_version("transformers") or runtime["safetensors"] != _package_version("safetensors"):
        raise oracle.OracleError("runtime package versions differ from validator environment")
    preflight = runtime["preflight"]
    expected_preflight = {"checkpoint_bytes", "headroom_factor", "mem_available_bytes", "mem_total_bytes", "required_headroom_bytes", "status"}
    if not isinstance(preflight, dict) or set(preflight) != expected_preflight:
        raise oracle.OracleError("runtime preflight keys differ")
    checkpoint_bytes = sum(entry["bytes"] for entry in manifest["identity"]["source_checkpoint"]["files"] if entry["file"].endswith(".safetensors"))
    if preflight["checkpoint_bytes"] != checkpoint_bytes or preflight["headroom_factor"] != 1.5 or preflight["required_headroom_bytes"] != int(checkpoint_bytes * 1.5) or preflight["status"] != "passed":
        raise oracle.OracleError("runtime preflight checkpoint arithmetic differs")
    if not isinstance(preflight["mem_available_bytes"], int) or not isinstance(preflight["mem_total_bytes"], int) or preflight["mem_available_bytes"] < preflight["required_headroom_bytes"] or preflight["mem_total_bytes"] < preflight["mem_available_bytes"]:
        raise oracle.OracleError("runtime preflight memory observation is invalid")
    run = runtime["run"]
    if not isinstance(run, dict) or set(run) != {"elapsed_seconds", "row_count"} or run["row_count"] != manifest["payload"]["record_count"] or isinstance(run["elapsed_seconds"], bool) or not isinstance(run["elapsed_seconds"], (int, float)) or not math.isfinite(run["elapsed_seconds"]) or run["elapsed_seconds"] <= 0:
        raise oracle.OracleError("runtime run summary differs")
    _validate_sha256s(root, manifest, context)
    if owned_context:
        context.verify_all()


def validate_oracle(
    root: Path,
    kind: str,
    *,
    context: oracle.ValidationContext | None = None,
    source_context: oracle.ValidationContext | None = None,
    verify_context: bool = True,
) -> dict[str, Any]:
    context = context or oracle.ValidationContext()
    source_context = source_context or oracle.ValidationContext()
    if root.is_symlink() or not root.is_dir():
        raise oracle.OracleError("oracle root must be a regular directory, not a symlink")
    manifest = oracle.validate_manifest(root, expected_kind=kind, context=context)
    blockers: list[str] = []
    if manifest["evidence_class"] == "synthetic_fixture":
        blockers.append("synthetic fixture is not an independent production oracle")
    if kind == "source" and manifest["identity"]["model_revision"] is None:
        blockers.append("source checkpoint revision metadata is unavailable or inconsistent")
    if kind == "path":
        if manifest["identity"]["artifact"]["package_manifest_sha256"] is None:
            raise oracle.OracleError("path oracle must bind a package manifest")
        artifact = manifest["identity"]["artifact"]
        runtime_path = root / "runtime.json"
        if runtime_path.exists() or runtime_path.is_symlink():
            _validate_path_runtime(root, manifest, context, source_context)
        elif manifest["evidence_class"] == "production":
            raise oracle.OracleError("production path oracle requires runtime.json")
        else:
            blockers.append("synthetic capture has no executable path runtime and is fixture-only")
        if artifact["artifact_manifest_sha256"] is None:
            package_ok, package_error = _package_binding_status(artifact, context)
            if not package_ok:
                blockers.append(package_error or "path oracle package binding is invalid")
    tokenizer = manifest["identity"]["tokenizer"]
    if {entry["file"] for entry in tokenizer["files"]} != set(oracle.TOKENIZER_FILES):
        raise oracle.OracleError("tokenizer file coverage differs")
    _rehash_files(tokenizer["root"], tokenizer["files"], "tokenizer")
    if kind == "source":
        source_checkpoint = manifest["identity"]["source_checkpoint"]
        _rehash_files(source_checkpoint["root"], source_checkpoint["files"], "source checkpoint")
        checkpoint_root = Path(source_checkpoint["root"])
        index = oracle.load_json(oracle.safe_relative(checkpoint_root, "model.safetensors.index.json", "source checkpoint index"))
        weight_map = index.get("weight_map") if isinstance(index, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise oracle.OracleError("source checkpoint index weight map is invalid")
        shards = set(weight_map.values())
        if len(shards) != 4 or any(not isinstance(name, str) for name in shards):
            raise oracle.OracleError("source checkpoint must contain exactly four indexed shards")
        expected_checkpoint_files = {"config.json", "model.safetensors.index.json", *shards}
        if {entry["file"] for entry in source_checkpoint["files"]} != expected_checkpoint_files:
            raise oracle.OracleError("source checkpoint file coverage differs")
        if manifest["evidence_class"] == "production":
            _validate_runtime(root, manifest, context)
    usable_key = "usable_as_source_evidence" if kind == "source" else "usable_as_path_evidence"
    usable = manifest[usable_key] and manifest["status"] == "available" and manifest["evidence_class"] == "production" and not blockers
    report = {
        "schema_version": "ullm.qwen35_aq4_p2_oracle_validator.v1",
        "status": "valid",
        "oracle_kind": manifest["oracle_kind"],
        "manifest_sha256": _sha(root / "manifest.json", context=context),
        "payload_sha256": manifest["payload"]["sha256"],
        "record_count": manifest["payload"]["record_count"],
        usable_key: usable,
        "promotion_eligible": False,
        "blockers": blockers,
    }
    if verify_context:
        context.verify_all()
        source_context.verify_all()
    return report


def validate_link(root: Path, source_root: Path, path_root: Path) -> dict[str, Any]:
    source_context = oracle.ValidationContext()
    path_context = oracle.ValidationContext()
    link_context = oracle.ValidationContext()
    source_report = validate_oracle(
        source_root, "source", context=source_context, verify_context=False
    )
    path_report = validate_oracle(
        path_root,
        "path",
        context=path_context,
        source_context=source_context,
        verify_context=False,
    )
    source = oracle.validate_manifest(
        source_root, expected_kind="source", context=source_context
    )
    path = oracle.validate_manifest(
        path_root, expected_kind="path", context=path_context
    )
    link = oracle.load_json(root / "manifest.json", context=link_context)
    if not isinstance(link, dict):
        raise oracle.OracleError("link manifest must be an object")
    expected = {"agreement", "created_utc", "evidence_class", "identity", "path", "promotion_eligible", "schema_version", "source", "status", "usable_as_p2_oracle_link"}
    if set(link) != expected:
        raise oracle.OracleError("link manifest keys differ")
    if link["schema_version"] != oracle.LINK_SCHEMA or link["status"] not in {"available", "fixture"}:
        raise oracle.OracleError("link schema or status is invalid")
    if link["evidence_class"] not in {"production", "synthetic_fixture"}:
        raise oracle.OracleError("link evidence_class is invalid")
    oracle.validate_utc(link["created_utc"])
    identity = link["identity"]
    if not isinstance(identity, dict) or set(identity) != {"model_id", "model_revision", "tokenizer_aggregate_sha256"}:
        raise oracle.OracleError("link identity keys differ")
    oracle.ensure_sha256(identity["tokenizer_aggregate_sha256"], "link tokenizer aggregate")
    if identity["model_id"] != source["identity"]["model_id"] or identity["model_id"] != path["identity"]["model_id"] or identity["model_revision"] != source["identity"]["model_revision"] or identity["model_revision"] != path["identity"]["model_revision"]:
        raise oracle.OracleError("link model identity differs")
    if identity["tokenizer_aggregate_sha256"] != source["identity"]["tokenizer"]["aggregate_sha256"] or identity["tokenizer_aggregate_sha256"] != path["identity"]["tokenizer"]["aggregate_sha256"]:
        raise oracle.OracleError("link tokenizer identity differs")
    for key, expected_root, manifest in (("source", source_root, source), ("path", path_root, path)):
        entry = link[key]
        if not isinstance(entry, dict) or set(entry) != ({"manifest_sha256", "payload_sha256"} if key == "source" else {"artifact_manifest_sha256", "manifest_sha256", "package_manifest_sha256", "payload_sha256"}):
            raise oracle.OracleError(f"link {key} keys differ")
        expected_context = source_context if key == "source" else path_context
        if entry["manifest_sha256"] != _sha(expected_root / "manifest.json", context=expected_context) or entry["payload_sha256"] != manifest["payload"]["sha256"]:
            raise oracle.OracleError(f"link {key} hash binding differs")
        if key == "path":
            if entry["artifact_manifest_sha256"] != manifest["identity"]["artifact"]["artifact_manifest_sha256"] or entry["package_manifest_sha256"] != manifest["identity"]["artifact"]["package_manifest_sha256"]:
                raise oracle.OracleError("link path artifact binding differs")
            if entry["artifact_manifest_sha256"] is not None:
                oracle.ensure_sha256(entry["artifact_manifest_sha256"], "link path artifact hash")
            oracle.ensure_sha256(entry["package_manifest_sha256"], "link path package hash")
    agreement = link["agreement"]
    if not isinstance(agreement, dict) or agreement != oracle.compare_payloads(
        source_root,
        source,
        path_root,
        path,
        source_context=source_context,
        path_context=path_context,
    ):
        raise oracle.OracleError("link agreement differs from bounded payload comparison")
    if link["promotion_eligible"] is not False:
        raise oracle.OracleError("source/path link must remain non-promotable until production policy accepts it")
    blockers = []
    if link["evidence_class"] == "synthetic_fixture":
        blockers.append("source/path link contains synthetic fixture evidence")
    if path["identity"]["artifact"]["artifact_manifest_sha256"] is None:
        package_ok, package_error = _package_binding_status(path["identity"]["artifact"], path_context)
        if not package_ok:
            blockers.append(package_error or "path oracle package binding is invalid")
    if not agreement["greedy_token_exact"] or not agreement["topk_exact"] or not agreement["hidden_sample_within_atol"] or not agreement["logit_sample_within_atol"]:
        blockers.append("source/path bounded agreement gate failed")
    if not agreement["hidden_sample_shape_exact"] or not agreement["logit_sample_shape_exact"]:
        blockers.append("source/path bounded sample shape differs")
    expected_usable = bool(link["evidence_class"] == "production" and source_report["usable_as_source_evidence"] and path_report["usable_as_path_evidence"] and not blockers)
    if link["usable_as_p2_oracle_link"] is not expected_usable:
        raise oracle.OracleError("link usable_as_p2_oracle_link differs from recomputed agreement")
    source_context.verify_all()
    path_context.verify_all()
    link_context.verify_all()
    return {"schema_version": "ullm.qwen35_aq4_p2_oracle_link_validator.v1", "status": "valid", "manifest_sha256": _sha(root / "manifest.json", context=link_context), "source_manifest_sha256": link["source"]["manifest_sha256"], "path_manifest_sha256": link["path"]["manifest_sha256"], "agreement": agreement, "usable_as_p2_oracle_link": expected_usable, "promotion_eligible": False, "blockers": blockers}


def probe_source(root: Path, payload: Path | None) -> dict[str, Any]:
    try:
        identity = oracle.inspect_source_model(root)
        source_status = "available"
        source_error = None
    except oracle.OracleError as error:
        identity = None
        source_status = "blocked"
        source_error = str(error)
    forward_status = "available" if payload is not None and payload.is_file() and not payload.is_symlink() else "blocked"
    blocker = None if forward_status == "available" else "independent BF16/F32 forward summaries are absent; checkpoint metadata alone is not an oracle"
    return {"schema_version": "ullm.qwen35_aq4_source_probe.v1", "status": "valid", "source_model": {"status": source_status, "identity": identity, "error": source_error}, "independent_forward_artifact": {"status": forward_status, "payload": str(payload) if payload else None, "blocker": blocker}, "production_oracle_available": source_status == "available" and forward_status == "available"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p_oracle = sub.add_parser("oracle")
    p_oracle.add_argument("root", type=Path)
    p_oracle.add_argument("--kind", choices=("source", "path"), required=True)
    p_oracle.add_argument("--output", type=Path)
    p_link = sub.add_parser("link")
    p_link.add_argument("root", type=Path)
    p_link.add_argument("--source-oracle", type=Path, required=True)
    p_link.add_argument("--path-oracle", type=Path, required=True)
    p_link.add_argument("--output", type=Path)
    p_probe = sub.add_parser("probe")
    p_probe.add_argument("--source-root", type=Path, required=True)
    p_probe.add_argument("--payload", type=Path)
    p_probe.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "oracle":
            report = validate_oracle(args.root, args.kind)
        elif args.command == "link":
            report = validate_link(args.root, args.source_oracle, args.path_oracle)
        else:
            report = probe_source(args.source_root, args.payload)
        raw = json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
        if args.output:
            if args.output.exists() or args.output.is_symlink():
                raise oracle.OracleError(f"refusing to overwrite report: {args.output}")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(raw, encoding="utf-8")
        else:
            print(raw, end="")
        return 0
    except (oracle.OracleError, OSError, ValueError) as error:
        print(f"Qwen3.5 AQ4 P2 oracle validation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
