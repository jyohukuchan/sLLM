#!/usr/bin/env python3
"""Compare one resident AQ4 worker with its legacy compatibility route.

The promotion receipt, copied profile, and served-model manifest used by this
smoke live only in a TemporaryDirectory.  This tool never creates or replaces
the product's promotion receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import select
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import ModuleType
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = ROOT / "deploy/served-models/qwen35-9b-aq4.profile.json"
DEFAULT_WORKER = ROOT / "target/release/ullm-aq4-worker"
DEFAULT_ENGINE = ROOT / "target/release/ullm-engine"
GENERATOR_PATH = ROOT / "tools/generate-served-model.py"
RESULT_SCHEMA = "ullm.aq4_resident_promotion_evidence.v1"
MAX_EVENT_BYTES = 4_194_304
STDERR_TAIL_BYTES = 65_536
LEGACY_PROFILE_ENVIRONMENT = (
    "ULLM_SERVED_MODEL_MANIFEST",
    "ULLM_MODEL_ID",
    "ULLM_MODEL_REVISION",
    "ULLM_ARTIFACT_CONTENT_SHA256",
    "ULLM_PACKAGE_MANIFEST_SHA256",
    "ULLM_DEVICE",
    "ULLM_EXECUTION_PROFILE",
    "ULLM_MODEL_CONTEXT_LENGTH",
    "ULLM_MAX_NEW_TOKENS",
    "ULLM_VOCAB_SIZE",
    "ULLM_EOS_TOKEN_IDS",
    "ULLM_TOP_K",
)
RAW_TOKEN_CASES = (
    {"id": "raw-p0001-g0004", "prompt_token_ids": [1], "max_new_tokens": 4},
    {
        "id": "raw-p0008-g0004",
        "prompt_token_ids": list(range(1, 9)),
        "max_new_tokens": 4,
    },
)
REASONING_CASE_ID = "reasoning-budget-zero"
PROMOTION_HIP_VISIBLE_DEVICES = "1"
PROMOTION_GPU_INDEX = "1"


class EvidenceError(RuntimeError):
    """Raised when promotion evidence is incomplete or inconsistent."""


def _exclusive_gpu_preflight() -> dict[str, Any]:
    """Prove that the deployed R9700 has no positive-VRAM KFD process."""

    try:
        result = subprocess.run(
            ["rocm-smi", "--showpids", "--json"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10.0,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise EvidenceError("ROCm GPU exclusivity preflight failed") from error
    if result.returncode != 0:
        raise EvidenceError("ROCm GPU exclusivity preflight returned nonzero")
    try:
        document = json.loads(result.stdout)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise EvidenceError("ROCm GPU exclusivity preflight returned invalid JSON") from error
    processes = document.get("system") if isinstance(document, dict) else None
    if not isinstance(processes, dict):
        raise EvidenceError("ROCm GPU exclusivity preflight lacks process data")

    positive_vram: list[dict[str, Any]] = []
    for pid, description in processes.items():
        if not isinstance(pid, str) or not pid.startswith("PID"):
            raise EvidenceError("ROCm GPU process identity is malformed")
        if not isinstance(description, str):
            raise EvidenceError("ROCm GPU process description is malformed")
        fields = [field.strip() for field in description.split(",")]
        if len(fields) < 3:
            raise EvidenceError("ROCm GPU process description is incomplete")
        try:
            vram_bytes = int(fields[2])
        except ValueError as error:
            raise EvidenceError("ROCm GPU process VRAM is not an integer") from error
        gpu_ids = {item.strip() for item in fields[1].split(",") if item.strip()}
        if PROMOTION_GPU_INDEX in gpu_ids and vram_bytes > 0:
            positive_vram.append(
                {
                    "pid": pid[3:],
                    "process": fields[0],
                    "gpu_index": PROMOTION_GPU_INDEX,
                    "vram_bytes": vram_bytes,
                }
            )
    if positive_vram:
        raise EvidenceError(
            "R9700 is not exclusive; positive-VRAM processes: "
            + json.dumps(positive_vram, sort_keys=True)
        )
    return {
        "tool": "rocm-smi --showpids --json",
        "gpu_index": PROMOTION_GPU_INDEX,
        "positive_vram_processes": [],
    }


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise EvidenceError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return module


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise EvidenceError(f"failed to read {label}: {path}") from error
    if not isinstance(value, dict):
        raise EvidenceError(f"{label} must be a JSON object")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    commit = result.stdout.strip()
    if result.returncode != 0 or not commit:
        raise EvidenceError("failed to resolve the source commit")
    return commit


def prepare_smoke_bundle(
    profile_path: Path,
    worker_binary: Path,
    temporary_root: Path,
    *,
    source_commit: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path]:
    """Create an ephemeral receipt/profile/manifest and return their documents."""

    profile = _read_object(profile_path, "AQ4 deployment profile")
    profile = json.loads(json.dumps(profile))
    promotion = profile.get("promotion")
    worker = profile.get("worker")
    if not isinstance(promotion, dict) or not isinstance(worker, dict):
        raise EvidenceError("AQ4 deployment profile lacks promotion or worker configuration")

    receipt = {"source_commit": source_commit}
    receipt_path = temporary_root / "smoke-promotion.json"
    profile_copy_path = temporary_root / "smoke-profile.json"
    manifest_path = temporary_root / "served-model.json"
    receipt_path.write_text(json.dumps(receipt, sort_keys=True) + "\n", encoding="ascii")
    promotion["receipt"] = os.fspath(receipt_path)
    promotion["source_commit_from_receipt"] = ["source_commit"]
    # Evidence must be produced before the production receipt exists.  The
    # ephemeral smoke manifest is therefore generated without the production
    # receipt/evidence gate; the final generator rebinds and validates it.
    promotion.pop("required_schema_version", None)
    promotion.pop("evidence_from_receipt", None)
    promotion.pop("evidence_sha256_from_receipt", None)
    worker["binary"] = os.fspath(worker_binary.resolve())
    profile_copy_path.write_text(
        json.dumps(profile, ensure_ascii=True, allow_nan=False, indent=2) + "\n",
        encoding="ascii",
    )

    generator = _load_module("_ullm_aq4_evidence_generator", GENERATOR_PATH)
    try:
        generator.generate(profile_copy_path, manifest_path)
    except Exception as error:
        raise EvidenceError(f"failed to generate ephemeral served-model manifest: {error}") from error
    manifest = _read_object(manifest_path, "ephemeral served-model manifest")
    return receipt, profile, manifest, manifest_path


def _worker_environment(manifest: dict[str, Any], *, legacy: bool) -> dict[str, str]:
    environment = dict(os.environ)
    # Match the deployed WRX80 isolation.  A standalone promotion command must
    # never inherit an unrestricted HIP device list from the caller shell.
    environment["HIP_VISIBLE_DEVICES"] = PROMOTION_HIP_VISIBLE_DEVICES
    for name in LEGACY_PROFILE_ENVIRONMENT:
        environment.pop(name, None)
    worker = manifest["worker"]
    environment["ULLM_WORKER_SCHEMA_VERSION"] = (
        "ullm.worker.v1" if legacy else str(worker["protocol"])
    )
    for name in worker["required_environment"]:
        environment[name] = "1"
    if legacy:
        public = manifest["public"]
        generation = manifest["generation"]
        product = manifest["product"]
        identity = worker["identity"]
        environment.update(
            {
                "ULLM_MODEL_ID": str(public["id"]),
                "ULLM_MODEL_REVISION": str(public["revision"]),
                "ULLM_ARTIFACT_CONTENT_SHA256": str(
                    (product.get("artifact") or product["package"])[
                        "content_sha256" if product.get("artifact") else "manifest_sha256"
                    ]
                ),
                "ULLM_PACKAGE_MANIFEST_SHA256": str(product["package"]["manifest_sha256"]),
                "ULLM_DEVICE": str(identity["device"]),
                "ULLM_EXECUTION_PROFILE": "rdna4_aq4_cli_compat",
                "ULLM_MODEL_CONTEXT_LENGTH": str(public["context_length"]),
                "ULLM_MAX_NEW_TOKENS": str(generation["max_completion_tokens"]),
                "ULLM_VOCAB_SIZE": str(generation["vocab_size"]),
                "ULLM_EOS_TOKEN_IDS": ",".join(map(str, generation["eos_token_ids"])),
                "ULLM_TOP_K": str(generation["sampling"]["top_k"]),
            }
        )
    return environment


def _read_event(process: subprocess.Popen[bytes], timeout_seconds: float) -> dict[str, Any]:
    assert process.stdout is not None
    ready, _, _ = select.select([process.stdout], [], [], timeout_seconds)
    if not ready:
        raise EvidenceError(f"worker event timed out after {timeout_seconds:.1f}s")
    line = process.stdout.readline(MAX_EVENT_BYTES + 2)
    if not line:
        raise EvidenceError(f"worker stdout closed unexpectedly (exit={process.poll()})")
    if len(line) > MAX_EVENT_BYTES + 1 or not line.endswith(b"\n"):
        raise EvidenceError("worker emitted an oversized or unterminated event")
    try:
        value = json.loads(line)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise EvidenceError("worker emitted invalid JSON") from error
    if not isinstance(value, dict):
        raise EvidenceError("worker event must be an object")
    return value


def _process_descendants(pid: int) -> list[dict[str, Any]]:
    pending = [pid]
    seen = {pid}
    descendants: list[dict[str, Any]] = []
    while pending:
        parent = pending.pop()
        children_path = Path(f"/proc/{parent}/task/{parent}/children")
        try:
            raw_children = children_path.read_text(encoding="ascii").split()
        except FileNotFoundError:
            continue
        except OSError as error:
            raise EvidenceError(f"failed to inspect worker descendants: {error}") from error
        for raw_pid in raw_children:
            child = int(raw_pid)
            if child in seen:
                continue
            seen.add(child)
            pending.append(child)
            comm_path = Path(f"/proc/{child}/comm")
            exe_path = Path(f"/proc/{child}/exe")
            try:
                comm = comm_path.read_text(encoding="utf-8").strip()
            except (OSError, UnicodeError):
                comm = "unavailable"
            try:
                executable = os.path.basename(os.readlink(exe_path))
            except OSError:
                executable = "unavailable"
            descendants.append({"pid": child, "comm": comm, "executable": executable})
    return sorted(descendants, key=lambda item: item["pid"])


def _inspect_resident_children(process: subprocess.Popen[bytes], phase: str) -> dict[str, Any]:
    descendants = _process_descendants(process.pid)
    sibling_engines = [
        item
        for item in descendants
        if item["comm"] == "ullm-engine" or item["executable"] == "ullm-engine"
    ]
    if sibling_engines:
        raise EvidenceError(f"resident worker spawned a sibling ullm-engine during {phase}")
    return {
        "phase": phase,
        "procfs_supported": Path("/proc").is_dir(),
        "descendants": descendants,
        "sibling_engine_count": 0,
    }


def _stderr_tail(handle: Any) -> str:
    handle.flush()
    size = handle.seek(0, os.SEEK_END)
    handle.seek(max(0, size - STDERR_TAIL_BYTES))
    return handle.read().decode("utf-8", errors="replace")


def _run_cases_in_process(
    command: list[str],
    manifest: dict[str, Any],
    *,
    mode: str,
    ready_timeout_seconds: float,
    request_timeout_seconds: float,
) -> dict[str, Any]:
    protocol = "ullm.worker.v1" if mode == "legacy" else str(manifest["worker"]["protocol"])
    if protocol not in {"ullm.worker.v1", "ullm.worker.v2"}:
        raise EvidenceError(f"{mode} worker protocol is unsupported: {protocol}")
    with tempfile.TemporaryFile(mode="w+b") as stderr:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=stderr,
            env=_worker_environment(manifest, legacy=mode == "legacy"),
            bufsize=0,
        )
        child_checks: list[dict[str, Any]] = []
        clean_shutdown = False
        try:
            ready = _read_event(process, ready_timeout_seconds)
            if ready.get("type") != "ready" or ready.get("schema_version") != protocol:
                raise EvidenceError(f"{mode} worker did not emit a valid ready event")
            if mode == "resident":
                child_checks.append(_inspect_resident_children(process, "ready"))

            cases_to_run = list(RAW_TOKEN_CASES)
            if mode == "resident" and protocol == "ullm.worker.v2":
                dialect = manifest.get("reasoning")
                if not isinstance(dialect, dict):
                    raise EvidenceError("v2 resident manifest has no reasoning dialect")
                cases_to_run.append(
                    {
                        "id": REASONING_CASE_ID,
                        "prompt_token_ids": [1],
                        "max_new_tokens": (
                            len(dialect["forced_end_token_ids"])
                            + int(dialect["reserved_answer_tokens"])
                        ),
                        "reasoning": {
                            "enabled": True,
                            "budget_tokens": 0,
                            "dialect_id": dialect["dialect_id"],
                            "end_token_ids": dialect["end_token_ids"],
                            "forced_end_token_ids": dialect["forced_end_token_ids"],
                            "reserved_answer_tokens": dialect["reserved_answer_tokens"],
                        },
                    }
                )

            results: list[dict[str, Any]] = []
            for case in cases_to_run:
                request_id = str(case["id"])
                command_record = {
                    "schema_version": protocol,
                    "type": "generate",
                    "request_id": request_id,
                    "prompt_token_ids": case["prompt_token_ids"],
                    "max_new_tokens": case["max_new_tokens"],
                    "sampling": {"temperature": 0.0, "top_p": 1.0, "top_k": 1, "seed": 0},
                    "eos_token_ids": manifest["generation"]["eos_token_ids"],
                }
                if "reasoning" in case:
                    command_record["reasoning"] = case["reasoning"]
                assert process.stdin is not None
                process.stdin.write(
                    json.dumps(command_record, separators=(",", ":")).encode("ascii") + b"\n"
                )
                process.stdin.flush()
                if mode == "resident":
                    child_checks.append(_inspect_resident_children(process, f"sent:{request_id}"))
                tokens: list[int] = []
                progress: list[int] = []
                started = False
                released: dict[str, Any] | None = None
                deadline = time.monotonic() + request_timeout_seconds
                while released is None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise EvidenceError(f"{mode} case {request_id} timed out")
                    event = _read_event(process, remaining)
                    if event.get("schema_version") != protocol:
                        raise EvidenceError(
                            f"{mode} case {request_id} emitted a mismatched schema version"
                        )
                    if mode == "resident":
                        child_checks.append(
                            _inspect_resident_children(
                                process, f"event:{request_id}:{event.get('type', 'unknown')}"
                            )
                        )
                    if event.get("type") == "error":
                        raise EvidenceError(f"{mode} case {request_id} failed: {event}")
                    if event.get("request_id") != request_id:
                        raise EvidenceError(f"{mode} worker emitted an event for another request")
                    event_type = event.get("type")
                    if event_type == "started":
                        if started:
                            raise EvidenceError(f"{mode} case {request_id} started twice")
                        started = True
                    elif event_type == "progress":
                        progress.append(int(event["processed_prompt_tokens"]))
                    elif event_type == "token":
                        if event.get("index") != len(tokens):
                            raise EvidenceError(f"{mode} case {request_id} token index is discontinuous")
                        tokens.append(int(event["token_id"]))
                    elif event_type == "released":
                        released = event
                    else:
                        raise EvidenceError(f"{mode} case {request_id} emitted unexpected event {event_type}")
                if not started or released.get("reset_complete") is not True:
                    raise EvidenceError(f"{mode} case {request_id} lacks start/reset evidence")
                if not isinstance(released.get("timings"), dict):
                    raise EvidenceError(f"{mode} case {request_id} lacks timing evidence")
                if released.get("completion_tokens") != len(tokens):
                    raise EvidenceError(f"{mode} case {request_id} completion count differs")
                reasoning_usage = None
                if "reasoning" in case:
                    reasoning_tokens = released.get("reasoning_tokens")
                    forced_end_tokens = released.get("forced_end_tokens")
                    if not isinstance(reasoning_tokens, int) or not isinstance(
                        forced_end_tokens, int
                    ):
                        raise EvidenceError(
                            f"{mode} case {request_id} lacks v2 reasoning usage"
                        )
                    if reasoning_tokens != 0:
                        raise EvidenceError(
                            f"{mode} case {request_id} budget-zero reasoning emitted tokens"
                        )
                    expected_forced = len(case["reasoning"]["forced_end_token_ids"])
                    if forced_end_tokens != expected_forced:
                        raise EvidenceError(
                            f"{mode} case {request_id} forced-end usage differs"
                        )
                    if reasoning_tokens + forced_end_tokens > len(tokens):
                        raise EvidenceError(
                            f"{mode} case {request_id} reasoning usage exceeds completion"
                        )
                    reasoning_usage = {
                        "reasoning_tokens": reasoning_tokens,
                        "forced_end_tokens": forced_end_tokens,
                    }
                if (
                    not progress
                    or progress[-1] != len(case["prompt_token_ids"])
                    or any(left >= right for left, right in zip(progress, progress[1:]))
                ):
                    raise EvidenceError(f"{mode} case {request_id} prompt progress is incomplete")
                results.append(
                    {
                        **case,
                        "tokens": tokens,
                        "outcome": released.get("outcome"),
                        "prompt_progress": progress,
                        "reset_complete": True,
                        "timings": released["timings"],
                        **({"reasoning_usage": reasoning_usage} if reasoning_usage else {}),
                    }
                )
                if mode == "resident":
                    child_checks.append(_inspect_resident_children(process, f"after:{request_id}"))

            assert process.stdin is not None
            process.stdin.write(
                json.dumps(
                    {"schema_version": protocol, "type": "shutdown"},
                    separators=(",", ":"),
                ).encode("ascii")
                + b"\n"
            )
            process.stdin.flush()
            process.stdin.close()
            return_code = process.wait(timeout=ready_timeout_seconds)
            if return_code != 0:
                raise EvidenceError(f"{mode} worker shutdown returned {return_code}")
            clean_shutdown = True
            return {
                "mode": mode,
                "pid": process.pid,
                "ready": ready,
                "cases": results,
                "child_process_checks": child_checks,
                "clean_shutdown": True,
                "stderr_tail": _stderr_tail(stderr),
            }
        finally:
            if not clean_shutdown and process.poll() is None:
                process.kill()
                process.wait(timeout=30)


def _atomic_write_json(path: Path, document: dict[str, Any]) -> None:
    if path.is_symlink():
        raise EvidenceError("output path must not be a symlink")
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(document, ensure_ascii=True, allow_nan=False, indent=2) + "\n").encode(
        "ascii"
    )
    temporary: Path | None = None
    try:
        descriptor, raw_path = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        temporary = Path(raw_path)
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(encoded)
            destination.flush()
            os.fsync(destination.fileno())
        temporary.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
        os.replace(temporary, path)
        temporary = None
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def run_evidence(
    profile_path: Path,
    output_path: Path,
    worker_binary: Path,
    legacy_engine: Path,
    *,
    ready_timeout_seconds: float,
    request_timeout_seconds: float,
    source_commit: str | None = None,
) -> dict[str, Any]:
    worker_binary = worker_binary.resolve()
    legacy_engine = legacy_engine.resolve()
    for executable, label in ((worker_binary, "AQ4 worker"), (legacy_engine, "legacy engine")):
        if not executable.is_file() or not os.access(executable, os.X_OK):
            raise EvidenceError(f"{label} is not executable: {executable}")
    commit = source_commit or _git_commit()
    gpu_preflight = _exclusive_gpu_preflight()

    with tempfile.TemporaryDirectory(prefix="ullm-aq4-promotion-evidence-") as raw_temporary:
        temporary = Path(raw_temporary)
        receipt, profile, manifest, manifest_path = prepare_smoke_bundle(
            profile_path, worker_binary, temporary, source_commit=commit
        )
        product_root = Path(manifest["product"]["root"])
        package_path = product_root / Path(manifest["product"]["package"]["manifest_path"]).parent

        resident = _run_cases_in_process(
            [os.fspath(worker_binary), "--served-model-manifest", os.fspath(manifest_path)],
            manifest,
            mode="resident",
            ready_timeout_seconds=ready_timeout_seconds,
            request_timeout_seconds=request_timeout_seconds,
        )
        # Deliberately start legacy only after resident has shut down and released GPU state.
        legacy = _run_cases_in_process(
            [
                os.fspath(worker_binary),
                "--engine",
                os.fspath(legacy_engine),
                "--package",
                os.fspath(package_path),
                "--device-index",
                "1",
                "--layers",
                "all",
            ],
            manifest,
            mode="legacy",
            ready_timeout_seconds=ready_timeout_seconds,
            request_timeout_seconds=request_timeout_seconds,
        )

        comparisons = []
        resident_raw_cases = [
            case for case in resident["cases"] if case["id"] != REASONING_CASE_ID
        ]
        for resident_case, legacy_case in zip(resident_raw_cases, legacy["cases"], strict=True):
            matches = resident_case["tokens"] == legacy_case["tokens"]
            comparisons.append({"id": resident_case["id"], "tokens_exact_match": matches})
            if not matches:
                raise EvidenceError(f"token mismatch for {resident_case['id']}")

        document = {
            "schema_version": RESULT_SCHEMA,
            "source_commit": commit,
            "created_at_unix_ns": time.time_ns(),
            "production_receipt_written": False,
            "gpu_exclusive_preflight": gpu_preflight,
            "ephemeral_bundle": {
                "receipt": receipt,
                "profile_sha256": hashlib.sha256(
                    (json.dumps(profile, ensure_ascii=True, allow_nan=False, indent=2) + "\n").encode(
                        "ascii"
                    )
                ).hexdigest(),
                "manifest_sha256": _sha256_file(manifest_path),
                "manifest": manifest,
            },
            "worker_binary": os.fspath(worker_binary),
            "worker_binary_sha256": _sha256_file(worker_binary),
            "legacy_engine": os.fspath(legacy_engine),
            "legacy_engine_sha256": _sha256_file(legacy_engine),
            "resident": resident,
            "legacy": legacy,
            "comparisons": comparisons,
            "verified": True,
        }
        _atomic_write_json(output_path, document)
        return document


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker-binary", type=Path, default=DEFAULT_WORKER)
    parser.add_argument("--legacy-engine", type=Path, default=DEFAULT_ENGINE)
    parser.add_argument("--ready-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--request-timeout-seconds", type=float, default=900.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        result = run_evidence(
            args.profile,
            args.output,
            args.worker_binary,
            args.legacy_engine,
            ready_timeout_seconds=args.ready_timeout_seconds,
            request_timeout_seconds=args.request_timeout_seconds,
        )
    except Exception as error:
        print(f"AQ4 resident promotion evidence failed: {error}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "schema_version": RESULT_SCHEMA,
                "output": os.fspath(args.output.resolve()),
                "manifest_sha256": result["ephemeral_bundle"]["manifest_sha256"],
                "verified": True,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
