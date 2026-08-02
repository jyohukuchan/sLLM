#!/usr/bin/env python3
"""Read-only Git index/tree hygiene check with the Phase 1 thresholds."""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ContractError, ROOT, identity, load_hygiene_allowlist  # noqa: E402

WARN_BLOB = 1 * 1024 * 1024
FAIL_BLOB = 10 * 1024 * 1024
FAIL_ADDED_BYTES = 50 * 1024 * 1024
WARN_NEW_PATHS = 200
FAIL_NEW_PATHS = 500


def git(args: list[str], repo: Path) -> str:
    proc = subprocess.run(["git", *args], cwd=repo, text=True, capture_output=True, check=False)
    if proc.returncode:
        raise ContractError(proc.stderr.strip() or f"git {' '.join(args)} failed")
    return proc.stdout


def index_paths(repo: Path) -> list[str]:
    raw = git(["ls-files", "-z"], repo)
    return sorted(item for item in raw.split("\0") if item)


def tree_paths(repo: Path, revision: str) -> list[str]:
    raw = git(["ls-tree", "-r", "--name-only", "-z", revision], repo)
    return sorted(item for item in raw.split("\0") if item)


def blob_size(repo: Path, path: str) -> int:
    for object_name in (f":{path}", f"HEAD:{path}"):
        proc = subprocess.run(["git", "cat-file", "-s", object_name], cwd=repo, text=True, capture_output=True, check=False)
        if proc.returncode == 0:
            return int(proc.stdout.strip())
    return 0


def allowlisted(path: str, entries: list[dict[str, object]]) -> bool:
    return any(fnmatch.fnmatchcase(path, str(entry["path"])) for entry in entries)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--base", default=os.environ.get("ULLM_H0_BASE_SHA"))
    args = parser.parse_args(argv)
    repo = args.repo.resolve()
    errors: list[str] = []
    warnings: list[str] = []
    try:
        paths = index_paths(repo)
        entries = load_hygiene_allowlist(repo)
        sizes = {path: blob_size(repo, path) for path in paths}
        base = args.base or "HEAD"
        modified = [line for line in git(["diff", "--name-only", base, "--"], repo).splitlines() if line]
        added = [line for line in git(["diff", "--diff-filter=A", "--name-only", base, "--"], repo).splitlines() if line]
        base_paths = tree_paths(repo, base)
        net_new_paths = max(0, len(paths) - len(base_paths))
        added_bytes = sum(sizes.get(path, 0) for path in added)
        prohibited_prefixes = (".local-artifacts/", "tests/fixtures/generated/", "trace/", "traces/", "profiles/", "benchmarks/raw/", "model/", "models/")
        prohibited = [path for path in paths if path.startswith(prohibited_prefixes) and not allowlisted(path, entries)]
        errors.extend(f"prohibited tracked path: {path}" for path in prohibited)
        over_fail = [f"{path}={size}" for path, size in sizes.items() if size > FAIL_BLOB and not allowlisted(path, entries)]
        over_warn = [f"{path}={size}" for path, size in sizes.items() if WARN_BLOB < size <= FAIL_BLOB and not allowlisted(path, entries)]
        errors.extend(f"tracked blob over 10 MiB: {item}" for item in over_fail)
        warnings.extend(f"tracked blob over 1 MiB: {item}" for item in over_warn)
        if added_bytes > FAIL_ADDED_BYTES:
            errors.append(f"one change adds more than 50 MiB of tracked content: {added_bytes} bytes")
        if len(added) > FAIL_NEW_PATHS:
            errors.append(f"one change adds more than 500 tracked paths: {len(added)}")
        elif len(added) > WARN_NEW_PATHS:
            warnings.append(f"one change adds more than 200 tracked paths: {len(added)}")
        if net_new_paths > FAIL_NEW_PATHS:
            errors.append(f"tracked path net increase exceeds 500: {net_new_paths}")
        summary = {
            "schema_version": "tracked-tree-v1",
            "tree_oid": identity(repo)["tree"],
            "base_revision": base,
            "tracked_count": len(paths),
            "modified_count": len(modified),
            "new_tracked_count": len(added),
            "net_new_tracked_count": net_new_paths,
            "tracked_bytes": sum(sizes.values()),
            "added_bytes": added_bytes,
            "largest_tracked": sorted(({"path": path, "bytes": size} for path, size in sizes.items()), key=lambda item: (-item["bytes"], item["path"]))[:20],
            "warnings": warnings,
            "errors": errors,
        }
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
        return 1 if errors else 0
    except (ContractError, OSError, ValueError) as exc:
        print(f"tracked tree: FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
