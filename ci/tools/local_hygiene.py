#!/usr/bin/env python3
"""Read-only local workspace hygiene report; this command never deletes data."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ContractError, ROOT, identity, load_hygiene_allowlist  # noqa: E402

WARN_UNTRACKED = 256 * 1024 * 1024
FAIL_UNTRACKED = 1 * 1024 * 1024 * 1024
WARN_IGNORED = 10 * 1024 * 1024 * 1024
FAIL_IGNORED = 20 * 1024 * 1024 * 1024
WARN_CHECKOUT = 20 * 1024 * 1024 * 1024
FAIL_CHECKOUT = 30 * 1024 * 1024 * 1024


def git(args: list[str], repo: Path) -> str:
    proc = subprocess.run(["git", *args], cwd=repo, text=True, capture_output=True, check=False)
    return proc.stdout if proc.returncode == 0 else ""


def git_paths(args: list[str], repo: Path) -> set[str]:
    proc = subprocess.run(["git", *args, "-z"], cwd=repo, capture_output=True, check=False)
    if proc.returncode != 0:
        return set()
    return {item.decode("utf-8", "surrogateescape") for item in proc.stdout.split(b"\0") if item}


def file_stats(repo: Path) -> dict[str, int]:
    stats = {"untracked_bytes": 0, "ignored_bytes": 0, "checkout_bytes": 0, "untracked_count": 0, "ignored_count": 0, "checkout_count": 0}
    top: dict[str, int] = {}
    untracked = git_paths(["ls-files", "--others", "--exclude-standard"], repo)
    ignored = git_paths(["ls-files", "--others", "--ignored", "--exclude-standard"], repo)
    for directory, names, filenames in os.walk(repo):
        names[:] = [name for name in names if name != ".git"]
        for filename in filenames:
            path = Path(directory) / filename
            try:
                size = path.stat().st_size
            except OSError:
                continue
            relative = path.relative_to(repo).as_posix()
            stats["checkout_bytes"] += size
            stats["checkout_count"] += 1
            top_key = relative.split("/", 1)[0]
            top[top_key] = top.get(top_key, 0) + size
            if relative in ignored:
                stats["ignored_bytes"] += size
                stats["ignored_count"] += 1
            elif relative in untracked:
                stats["untracked_bytes"] += size
                stats["untracked_count"] += 1
    stats["top_directories"] = sorted(top.items(), key=lambda item: (-item[1], item[0]))[:20]  # type: ignore[assignment]
    return stats


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    repo = args.repo.resolve()
    try:
        allowlist_entries = load_hygiene_allowlist(repo)
        stats = file_stats(repo)
    except (ContractError, OSError, ValueError) as exc:
        print(f"local hygiene: FAIL: {exc}", file=sys.stderr)
        return 1
    warnings: list[str] = []
    errors: list[str] = []
    for key, warning, failure in (("untracked_bytes", WARN_UNTRACKED, FAIL_UNTRACKED), ("ignored_bytes", WARN_IGNORED, FAIL_IGNORED), ("checkout_bytes", WARN_CHECKOUT, FAIL_CHECKOUT)):
        if stats[key] > failure:
            errors.append(f"{key} exceeds stop threshold: {stats[key]}")
        elif stats[key] > warning:
            warnings.append(f"{key} exceeds warning threshold: {stats[key]}")
    worktree_text = git(["worktree", "list", "--porcelain"], repo)
    worktree_records: list[dict[str, object]] = []
    now = int(time.time())
    for block in worktree_text.strip().split("\n\n") if worktree_text.strip() else []:
        fields: dict[str, str | bool] = {}
        for line in block.splitlines():
            key, _, value = line.partition(" ")
            fields[key] = value if value else True
        path = str(fields.get("worktree", ""))
        branch = str(fields.get("branch", "")).removeprefix("refs/heads/") or None
        exists = bool(path) and Path(path).exists()
        dirty = bool(git(["-C", path, "status", "--porcelain"], repo).strip()) if exists else None
        timestamp_text = git(["-C", path, "log", "-1", "--format=%ct"], repo).strip() if exists else ""
        timestamp = int(timestamp_text) if timestamp_text.isdigit() else None
        age_days = (now - timestamp) / 86400 if timestamp is not None else None
        locked = "locked" in fields
        stale = bool(exists and branch != "main" and dirty is False and not locked and age_days is not None and age_days > 14)
        worktree_records.append({"path": path, "branch": branch, "exists": exists, "dirty": dirty, "locked": locked, "last_activity_unix": timestamp, "age_days": age_days, "stale_candidate": stale})
    worktrees = [str(record["path"]) for record in worktree_records]
    missing_worktrees = [path for path in worktrees if not Path(path).exists()]
    errors.extend(f"registered worktree path is missing: {path}" for path in missing_worktrees)
    if len(worktrees) > 4:
        errors.append(f"registered worktrees exceed 4: {len(worktrees)}")
    elif len(worktrees) > 3:
        warnings.append(f"registered worktrees exceed 3: {len(worktrees)}")
    prune = git(["worktree", "prune", "--dry-run", "--verbose"], repo).splitlines()
    upstream = git(["rev-list", "--left-right", "--count", "@{upstream}...HEAD"], repo).split()
    ahead = int(upstream[1]) if len(upstream) == 2 and upstream[1].isdigit() else None
    behind = int(upstream[0]) if len(upstream) == 2 and upstream[0].isdigit() else None
    if ahead is not None and ahead > 20:
        errors.append(f"branch is more than 20 commits ahead of upstream: {ahead}")
    branch = git(["branch", "--show-current"], repo).strip() or None
    upstream_name = git(["rev-parse", "--abbrev-ref", "@{upstream}"], repo).strip() or None
    last_activity_text = git(["log", "-1", "--format=%ct"], repo).strip()
    last_activity = int(last_activity_text) if last_activity_text.isdigit() else None
    if upstream_name is None:
        warnings.append("current branch has no upstream")
    if branch != "main" and ahead and last_activity is not None and (now - last_activity) > 7 * 86400:
        errors.append("feature branch has unpushed commits and no activity for more than 7 days")
    report = {
        "schema_version": "local-hygiene-v1",
        "tree_oid": identity(repo)["tree"],
        "sizes_counts": stats,
        "worktrees": {"count": len(worktrees), "entries": worktree_records, "stale_registration_candidates": prune},
        "remote_sync": {"branch": branch, "upstream": upstream_name, "ahead": ahead, "behind": behind, "last_activity_unix": last_activity},
        "allowlist": {"entries": len(allowlist_entries), "validated": True},
        "warnings": warnings,
        "errors": errors,
        "read_only": True,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
