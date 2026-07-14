#!/usr/bin/env python3
"""Run the dedicated CPU AQ4 all-M=1 path oracle and capture bounded evidence.

The source oracle supplies only the replay token sequence (the source payload is
never copied into the path payload).  The Rust binary performs one model load,
streams bounded observations, and emits JSONL.  This bridge validates that
stream, binds the real package manifest, and optionally creates the source/path
link.  Products with no separate artifact manifest must opt into the explicit
package-only identity mode; they remain non-usable for promotion.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TOOLS = Path(__file__).resolve().parent


def _load_tool(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load tool {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CAPTURE = _load_tool("capture_qwen35_aq4_p2_oracle", "capture-qwen35-aq4-p2-oracle.py")
VALIDATE = _load_tool("validate_qwen35_aq4_p2_oracle", "validate-qwen35-aq4-p2-oracle.py")
ORACLE = CAPTURE.oracle

MAX_STDOUT_BYTES = ORACLE.MAX_PAYLOAD_BYTES
MAX_STDERR_BYTES = 1 * 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 4 * 60 * 60
DEFAULT_SERVED_MODEL_MANIFEST = Path("/etc/ullm/served-models/active.json")
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


def _regular(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_file():
        raise ORACLE.OracleError(f"{label} must be a regular non-symlink file")
    return path


def _directory(path: Path, label: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise ORACLE.OracleError(f"{label} must be a regular non-symlink directory")
    return path


def _sha(path: Path) -> str:
    return ORACLE.sha256_file(_regular(path, "hashed file"))


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":")) + "\n").encode("ascii")


def _load_source(source_root: Path, cases_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, list[int]]]:
    source = VALIDATE.validate_oracle(source_root, "source")
    source_manifest = ORACLE.validate_manifest(source_root, expected_kind="independent_source")
    cases = CAPTURE._load_cases(cases_path)
    if source_manifest["cases"] != cases:
        raise ORACLE.OracleError("source oracle cases differ from cases JSON")
    by_case: dict[str, list[int]] = {case["case_id"]: [] for case in cases}
    for record in ORACLE.payload_records(source_root, source_manifest):
        by_case[record["case_id"]].append(record["greedy_token_id"])
    for case in cases:
        token_ids = by_case[case["case_id"]]
        if len(token_ids) != case["step_count"]:
            raise ORACLE.OracleError(f"source replay row count differs for {case['case_id']}")
    return source_manifest, cases, by_case


def _guard_environment(manifest_path: Path | None, *, production: bool) -> tuple[dict[str, str], dict[str, Any]]:
    if manifest_path is None:
        if production:
            raise ORACLE.OracleError("production path oracle requires --served-model-manifest")
        return dict(os.environ), {"manifest": None, "manifest_sha256": None, "required_environment": [], "required_environment_sha256": None}
    manifest = ORACLE.load_json(_regular(manifest_path, "served-model manifest"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("worker"), dict):
        raise ORACLE.OracleError("served-model manifest worker object is invalid")
    names = manifest["worker"].get("required_environment")
    if not isinstance(names, list) or any(not isinstance(name, str) for name in names):
        raise ORACLE.OracleError("served-model required_environment is invalid")
    expected = set(REQUIRED_HIP_KERNEL_ENV)
    actual = set(names)
    if len(names) != len(actual) or actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ORACLE.OracleError(f"served-model required_environment differs: missing={missing} extra={extra}")
    inherited = {name: value for name, value in os.environ.items() if name.startswith("ULLM_REQUIRE_HIP_")}
    bad = {name: value for name, value in inherited.items() if value != "1"}
    extra = set(inherited) - expected
    if bad or extra:
        raise ORACLE.OracleError(f"inherited HIP guard environment is not exact: bad={bad} extra={sorted(extra)}")
    child_env = dict(os.environ)
    for name in list(child_env):
        if name.startswith("ULLM_REQUIRE_HIP_"):
            del child_env[name]
    for name in REQUIRED_HIP_KERNEL_ENV:
        child_env[name] = "1"
    return child_env, {
        "manifest": str(manifest_path.resolve(strict=True)),
        "manifest_sha256": _sha(manifest_path),
        "required_environment": list(REQUIRED_HIP_KERNEL_ENV),
        "required_environment_sha256": ORACLE.canonical_sha256(list(REQUIRED_HIP_KERNEL_ENV)),
    }


def _write_replay(path: Path, cases: list[dict[str, Any]], by_case: dict[str, list[int]]) -> None:
    value = {"cases": [{"case_id": case["case_id"], "token_ids": by_case[case["case_id"]]} for case in cases]}
    path.write_bytes(_canonical(value))


def _run_binary(
    binary: Path,
    package_dir: Path,
    cases: Path,
    replay: Path,
    *,
    device_index: int,
    chunk_bytes: int,
    prefill_m: int,
    rotary_dim: int | None,
    rope_base: float | None,
    timeout_seconds: float,
    environment: dict[str, str],
) -> tuple[bytes, bytes, float]:
    _regular(binary, "path oracle binary")
    _directory(package_dir, "package directory")
    command = [
        str(binary),
        str(package_dir),
        str(cases),
        str(replay),
        str(device_index),
        str(chunk_bytes),
        str(prefill_m),
    ]
    if rotary_dim is not None:
        command.append(str(rotary_dim))
    if rope_base is not None:
        command.append(str(rope_base))
    started = time.monotonic()
    try:
        result = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout_seconds, env=environment)
    except subprocess.TimeoutExpired as error:
        raise ORACLE.OracleError(f"path oracle timed out after {timeout_seconds:g}s") from error
    elapsed = time.monotonic() - started
    if len(result.stdout) > MAX_STDOUT_BYTES:
        raise ORACLE.OracleError("path oracle stdout exceeds bounded payload limit")
    if len(result.stderr) > MAX_STDERR_BYTES:
        raise ORACLE.OracleError("path oracle stderr exceeds bounded diagnostic limit")
    if result.returncode != 0:
        diagnostic = result.stderr.decode("utf-8", errors="replace")[-4096:]
        raise ORACLE.OracleError(f"path oracle binary failed with exit {result.returncode}: {diagnostic}")
    if not result.stdout.strip():
        raise ORACLE.OracleError("path oracle binary emitted no rows")
    return result.stdout, result.stderr, elapsed


def _canonicalize_payload(raw: bytes, path: Path, cases: list[dict[str, Any]]) -> tuple[int, str]:
    if len(raw) == 0 or len(raw) > MAX_STDOUT_BYTES:
        raise ORACLE.OracleError("path oracle output is empty or exceeds bounded payload limit")
    try:
        path.write_bytes(raw)
    except OSError as error:
        raise ORACLE.OracleError(f"cannot stage path payload: {error}") from error
    with tempfile.NamedTemporaryFile(prefix="qwen35-aq4-path-canonical-", suffix=".jsonl", delete=False) as handle:
        canonical_path = Path(handle.name)
    try:
        digest, _, records = CAPTURE._copy_payload(path, canonical_path, cases)
        os.replace(canonical_path, path)
    finally:
        if canonical_path.exists():
            canonical_path.unlink()
    return records, digest


def _write_runtime(output: Path, *, package_dir: Path, package_manifest: Path, artifact_manifest: Path | None, binary: Path, source_root: Path, source_manifest: dict[str, Any], row_count: int, elapsed_seconds: float, guard_identity: dict[str, Any]) -> None:
    runtime = {
        "schema_version": "ullm.qwen35_aq4_path_oracle_runtime.v1",
        "runtime": "ullm-aq4-p2-path-oracle",
        "device": "cpu",
        "dtype": "f32",
        "all_m1": True,
        "model_loads": 1,
        "package_dir": str(package_dir.resolve(strict=True)),
        "package_manifest": str(package_manifest.resolve(strict=True)),
        "package_manifest_sha256": _sha(package_manifest),
        "artifact_manifest": str(artifact_manifest.resolve(strict=True)) if artifact_manifest is not None else None,
        "artifact_manifest_sha256": _sha(artifact_manifest) if artifact_manifest is not None else None,
        "binary": {"path": str(binary.resolve(strict=True)), "sha256": _sha(binary)},
        "source_replay": {
            "manifest_sha256": _sha(source_root / "manifest.json"),
            "payload_sha256": source_manifest["payload"]["sha256"],
        },
        "served_model_guard": guard_identity,
        "run": {"elapsed_seconds": elapsed_seconds, "row_count": row_count},
    }
    path = output / "runtime.json"
    if os.path.lexists(path):
        raise ORACLE.OracleError(f"refusing to overwrite runtime sidecar: {path}")
    path.write_bytes(_canonical(runtime))


def _write_sums(output: Path) -> None:
    sums = []
    for name in ("manifest.json", "payload.jsonl", "runtime.json"):
        sums.append(f"{_sha(output / name)}  {name}")
    path = output / "SHA256SUMS"
    if os.path.lexists(path):
        raise ORACLE.OracleError(f"refusing to overwrite checksum sidecar: {path}")
    path.write_text("\n".join(sums) + "\n", encoding="ascii")


def export(args: argparse.Namespace) -> dict[str, Any]:
    package_dir = _directory(args.package_dir, "package directory")
    package_manifest = _regular(args.package_manifest, "package manifest")
    try:
        package_manifest.resolve(strict=True).relative_to(package_dir.resolve(strict=True))
    except ValueError as error:
        raise ORACLE.OracleError("package manifest must be inside package directory") from error
    artifact_manifest = _regular(args.artifact_manifest, "artifact manifest") if args.artifact_manifest is not None else None
    if artifact_manifest is not None and artifact_manifest.resolve() == package_manifest.resolve():
        raise ORACLE.OracleError("artifact and package manifests must be distinct files")
    if artifact_manifest is None and not args.allow_package_only:
        raise ORACLE.OracleError("missing artifact manifest; pass --allow-package-only for package-only products")
    environment, guard_identity = _guard_environment(args.served_model_manifest, production=args.evidence_class == "production")
    source_manifest, cases, replay_by_case = _load_source(args.source_oracle, args.cases)
    with tempfile.TemporaryDirectory(prefix="qwen35-aq4-path-oracle-") as temporary:
        temporary_root = Path(temporary)
        replay_path = temporary_root / "replay.json"
        _write_replay(replay_path, cases, replay_by_case)
        stdout, _, elapsed = _run_binary(
            args.binary,
            package_dir,
            args.cases,
            replay_path,
            device_index=args.device_index,
            chunk_bytes=args.chunk_bytes,
            prefill_m=args.prefill_m,
            rotary_dim=args.rotary_dim,
            rope_base=args.rope_base,
            timeout_seconds=args.timeout_seconds,
            environment=environment,
        )
        payload_path = temporary_root / "payload.jsonl"
        records, payload_sha = _canonicalize_payload(stdout, payload_path, cases)
        capture_args = argparse.Namespace(
            output=args.output,
            cases=args.cases,
            payload=payload_path,
            kind="path",
            evidence_class=args.evidence_class,
            source_root=None,
            tokenizer_root=args.tokenizer_root,
            tokenizer_file=list(ORACLE.TOKENIZER_FILES),
            artifact_manifest=artifact_manifest,
            package_manifest=package_manifest,
            model_id=args.model_id or source_manifest["identity"]["model_id"],
            model_revision=args.model_revision if args.model_revision is not None else source_manifest["identity"]["model_revision"],
        )
        manifest = CAPTURE.capture(capture_args)
        _write_runtime(
            args.output,
            package_dir=package_dir,
            package_manifest=package_manifest,
            artifact_manifest=artifact_manifest,
            binary=args.binary,
            source_root=args.source_oracle,
            source_manifest=source_manifest,
            row_count=records,
            elapsed_seconds=elapsed,
            guard_identity=guard_identity,
        )
        _write_sums(args.output)
    report = VALIDATE.validate_oracle(args.output, "path")
    result: dict[str, Any] = {"path": report, "payload_sha256": payload_sha, "runtime": ORACLE.load_json(args.output / "runtime.json")}
    if args.link_output is not None:
        link_args = argparse.Namespace(source_oracle=args.source_oracle, path_oracle=args.output, output=args.link_output)
        link_manifest = CAPTURE.link(link_args)
        result["link"] = VALIDATE.validate_link(args.link_output, args.source_oracle, args.output)
        result["link_manifest"] = link_manifest
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--package-manifest", type=Path, required=True)
    parser.add_argument("--artifact-manifest", type=Path)
    parser.add_argument("--allow-package-only", action="store_true")
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--source-oracle", type=Path, required=True)
    parser.add_argument("--tokenizer-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--link-output", type=Path)
    parser.add_argument("--binary", type=Path, default=ROOT / "target/debug/ullm-aq4-p2-path-oracle")
    parser.add_argument("--served-model-manifest", type=Path, default=DEFAULT_SERVED_MODEL_MANIFEST)
    parser.add_argument("--model-id")
    parser.add_argument("--model-revision")
    parser.add_argument("--evidence-class", choices=("production", "synthetic_fixture"), default="production")
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--chunk-bytes", type=int, default=1024 * 1024)
    parser.add_argument("--prefill-m", type=int, default=1)
    parser.add_argument("--rotary-dim", type=int, default=64)
    parser.add_argument("--rope-base", type=float, default=10_000_000.0)
    parser.add_argument("--timeout-seconds", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    args = parser.parse_args(argv)
    try:
        if args.device_index < 0 or args.chunk_bytes <= 0 or args.prefill_m <= 0 or args.rotary_dim <= 0 or args.rope_base <= 0 or args.timeout_seconds <= 0:
            raise ORACLE.OracleError("numeric execution options must be positive")
        result = export(args)
        print(json.dumps(result, ensure_ascii=True, sort_keys=True))
        return 0
    except (ORACLE.OracleError, OSError, ValueError, subprocess.SubprocessError) as error:
        print(f"Qwen3.5 AQ4 P2 path oracle export failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
