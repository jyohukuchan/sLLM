#!/usr/bin/env python3
"""Reconstruct the pinned Stage 9 draft-vocabulary corpus manifest.

The source corpora and reference repositories stay outside Git.  This tool is
the small, tracked recipe which verifies their identities and recreates the
manifest consumed by :mod:`qwen38_draft_vocab`.  It never copies source text.

The lock records repository revisions, parquet identities, selection rules,
and digests of the selected source rows.  Code sources are selected from
``git ls-files`` in the fixed repository order.  Within each repository the
SHA-256 of ``repository/path`` provides a stable order; one path is taken from
each repository in every round until the category byte target is reached.
The historical Stage 9 manifest has one recorded deferred path in the Python
category; the lock makes that ordering quirk explicit and reproducible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "qwen38-draft-vocab-source-lock-v1"
MANIFEST_SCHEMA = "qwen38-draft-vocab-v1"
REPOSITORY_ORDER = ("vLLM", "SGLang", "llama.cpp", "KTransformers", "LMDeploy", "TensorRT-LLM")
DEFAULT_DATASET_ROOT = Path("/home/homelab1/datapool/dataset")
DEFAULT_MODEL_ROOT = Path("/home/homelab1/datapool/ai_models/safetensors/Qwen3.8-27B-NVFP4")
DEFAULT_MAX_SOURCE_BYTES = 200_000
DEFAULT_MAX_ROWS = 1_000_000


class SourceLockError(ValueError):
    """Raised when an input does not match the pinned source lock."""


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise SourceLockError(f"cannot read {path}: {exc}") from exc
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SourceLockError(f"cannot parse {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SourceLockError(f"{path} must contain an object")
    return value


def _git(repo: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", "")
        raise SourceLockError(f"git {' '.join(args)} failed in {repo}: {detail}") from exc
    return result.stdout


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path)


def _verify_repositories(lock: Mapping[str, Any], repo_root: Path) -> dict[str, Path]:
    repositories = lock.get("repositories")
    if not isinstance(repositories, list) or [item.get("name") for item in repositories] != list(REPOSITORY_ORDER):
        raise SourceLockError("repositories must use the pinned repository order")
    result: dict[str, Path] = {}
    for item in repositories:
        if not isinstance(item, dict):
            raise SourceLockError("repository entries must be objects")
        name = item.get("name")
        path = _resolve(repo_root, item.get("path", ""))
        if name in result or not path.is_dir():
            raise SourceLockError(f"repository path is unavailable: {name} {path}")
        head = _git(path, "rev-parse", "HEAD").strip()
        if head != item.get("head"):
            raise SourceLockError(f"{name} HEAD mismatch: expected {item.get('head')}, got {head}")
        status = _git(path, "status", "--porcelain=v1", "--untracked-files=all")
        if status:
            raise SourceLockError(f"{name} has a dirty tree:\n{status}")
        result[name] = path
    return result


def _verify_pinned_file(root: Path, item: Mapping[str, Any], label: str) -> Path:
    relative = item.get("path")
    if not isinstance(relative, str) or not relative:
        raise SourceLockError(f"{label}.path must be a non-empty string")
    path = _resolve(root, relative).resolve()
    expected = item.get("sha256")
    if not isinstance(expected, str) or len(expected) != 64:
        raise SourceLockError(f"{label}.sha256 is invalid")
    actual = sha256_file(path)
    if actual != expected:
        raise SourceLockError(f"{label} SHA mismatch: expected {expected}, got {actual}")
    return path


def _verify_parquet_sources(lock: Mapping[str, Any], dataset_root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    sources = lock.get("parquet_sources")
    if not isinstance(sources, list) or len(sources) != 4:
        raise SourceLockError("exactly four parquet_sources are required")
    for item in sources:
        if not isinstance(item, dict):
            raise SourceLockError("parquet source entries must be objects")
        key = item.get("domain")
        if not isinstance(key, str) or key in result:
            raise SourceLockError("parquet source domain is missing or duplicated")
        path = _verify_pinned_file(dataset_root, item, f"parquet_sources[{key}]")
        if item.get("format") != "parquet":
            raise SourceLockError(f"{key} must be parquet")
        result[key] = path
    return result


def _git_files(repo: Path) -> list[str]:
    raw = _git(repo, "ls-files", "-z").encode("utf-8", "surrogateescape")
    return [item.decode("utf-8", "strict") for item in raw.split(b"\0") if item]


def _candidate_paths(repo_name: str, repo: Path, category: Mapping[str, Any]) -> list[dict[str, Any]]:
    suffixes = set(category["suffixes"])
    max_bytes = int(category["max_source_bytes"])
    candidates: list[dict[str, Any]] = []
    for relative in _git_files(repo):
        if Path(relative).suffix.lower() not in suffixes:
            continue
        path = repo / relative
        if not path.is_file():
            continue
        try:
            data = path.read_bytes()
            data.decode("utf-8", "strict")
        except UnicodeError:
            continue
        except OSError as exc:
            raise SourceLockError(f"cannot read source {path}: {exc}") from exc
        if len(data) > max_bytes or not data.strip():
            continue
        candidates.append(
            {
                "dataset": repo_name,
                "revision": category["repository_revision"],
                "file": relative,
                "path": path.resolve(),
                "sha256": sha256_bytes(data),
                "format": "text",
                "max_bytes": len(data),
                "max_rows": int(category["max_rows"]),
            }
        )
    candidates.sort(key=lambda item: sha256_bytes(f"{repo_name}/{item['file']}".encode("utf-8")))
    return candidates


def _selection_hash(files: Iterable[Mapping[str, Any]]) -> str:
    rows = [
        "\t".join(
            str(item[key])
            for key in ("dataset", "revision", "file", "sha256", "max_bytes")
        )
        for item in files
    ]
    return sha256_bytes("\n".join(rows).encode("utf-8"))


def _select_code_sources(lock: Mapping[str, Any], repositories: Mapping[str, Path], name: str) -> list[dict[str, Any]]:
    category = lock["code_categories"][name]
    lists = {
        repo: _candidate_paths(repo, repositories[repo], {**category, "repository_revision": next(
            item["head"] for item in lock["repositories"] if item["name"] == repo
        )})
        for repo in REPOSITORY_ORDER
    }
    target = int(category["target_bytes"])
    selected: list[dict[str, Any]] = []
    total = 0
    round_index = 0
    cursors = {repo: 0 for repo in REPOSITORY_ORDER}
    pending: dict[str, dict[str, Any]] = {}
    defer_rules = category.get("defer", [])
    while total < target:
        for repo in REPOSITORY_ORDER:
            item = pending.pop(repo, None)
            if item is None:
                index = cursors[repo]
                if index >= len(lists[repo]):
                    continue
                item = lists[repo][index]
                cursors[repo] += 1
            matching = [
                rule
                for rule in defer_rules
                if rule.get("round") == round_index and rule.get("repo") == repo and rule.get("file") == item["file"]
            ]
            if matching:
                pending[repo] = item
                continue
            selected.append(item)
            total += item["max_bytes"]
            if total >= target:
                break
        if total >= target:
            break
        round_index += 1
    for rule in defer_rules:
        if rule.get("round") is None:
            continue
        expected = (rule["repo"], rule["file"])
        found = next((i for i, item in enumerate(selected) if (item["dataset"], item["file"]) == expected), None)
        if found is None:
            raise SourceLockError(f"deferred source was not selected: {expected}")
    expected_count = int(category["selected_count"])
    expected_bytes = int(category["selected_bytes"])
    if len(selected) != expected_count or total != expected_bytes:
        raise SourceLockError(f"{name} selection size mismatch: {len(selected)}/{total}")
    digest = _selection_hash(selected)
    if digest != category["selected_file_hashes_sha256"]:
        raise SourceLockError(f"{name} selected file hash mismatch: {digest}")
    return selected


def build_manifest(lock: Mapping[str, Any], repo_root: Path, dataset_root: Path, model_root: Path) -> dict[str, Any]:
    if lock.get("schema_version") != SCHEMA_VERSION:
        raise SourceLockError(f"schema_version must be {SCHEMA_VERSION}")
    if lock.get("manifest_schema_version") != MANIFEST_SCHEMA:
        raise SourceLockError(f"manifest_schema_version must be {MANIFEST_SCHEMA}")
    repositories = _verify_repositories(lock, repo_root)
    parquet_paths = _verify_parquet_sources(lock, dataset_root)
    tokenizer = lock["tokenizer"]
    tokenizer_path = _verify_pinned_file(model_root, tokenizer, "tokenizer")
    domains: list[dict[str, Any]] = []
    for source in lock["parquet_sources"]:
        item = dict(source)
        item.pop("domain", None)
        item["path"] = str(parquet_paths[source["domain"]].resolve())
        domains.append({"name": source["domain"], "files": [item]})
    for name in ("code_py", "code_cpp", "code_other"):
        files = _select_code_sources(lock, repositories, name)
        domains.append({"name": name, "files": [{key: value for key, value in item.items() if key != "path"} | {"path": str(item["path"])} for item in files]})
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "tokenizer": {"path": str(tokenizer_path.resolve()), "sha256": tokenizer["sha256"]},
        "vocab_size": lock["vocab_size"],
        "required_special_tokens": lock["required_special_tokens"],
        "overlap_exclusions": lock["overlap_exclusions"],
        "domains": domains,
    }
    digest = sha256_bytes(canonical_json(manifest))
    if digest != lock["manifest_sha256"]:
        raise SourceLockError(f"canonical manifest SHA mismatch: expected {lock['manifest_sha256']}, got {digest}")
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, default=Path(__file__).resolve().parents[1] / "matrix/qwen38-draft-vocab-source-lock-v1.json")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2] / "reference")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidate-vocab", type=Path, help="optional uint32 output to verify against the lock")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    lock = _read_json(args.lock.resolve())
    manifest = build_manifest(lock, args.repo_root.resolve(), args.dataset_root.resolve(), args.model_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.candidate_vocab:
        digest = sha256_file(args.candidate_vocab.resolve())
        expected = lock.get("candidate_vocab_sha256")
        if digest != expected:
            raise SourceLockError(f"candidate vocabulary SHA mismatch: expected {expected}, got {digest}")
    print(f"manifest_sha256={lock['manifest_sha256']}")
    print(f"output={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SourceLockError as exc:
        raise SystemExit(f"qwen38_draft_vocab_sources: error: {exc}")
