#!/usr/bin/env python3
"""Root-only, fixed-path lifecycle helper for the SQ8 promotion device lock."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from pathlib import Path


RUNTIME_DIR = Path("/run/ullm")
LOCK_PATH = RUNTIME_DIR / "device-1.lock"
OWNER_UID = 1000
OWNER_GID = 1000


class LockHelperError(RuntimeError):
    pass


def _metadata(path: Path) -> dict[str, int | str]:
    value = path.stat(follow_symlinks=False)
    return {
        "path": str(path),
        "device": value.st_dev,
        "inode": value.st_ino,
        "mode": f"{stat.S_IMODE(value.st_mode):04o}",
        "uid": value.st_uid,
        "gid": value.st_gid,
        "nlink": value.st_nlink,
    }


def _validate_regular(path: Path, *, device: int | None = None, inode: int | None = None) -> os.stat_result:
    value = path.stat(follow_symlinks=False)
    if (
        path.is_symlink()
        or not stat.S_ISREG(value.st_mode)
        or stat.S_IMODE(value.st_mode) != 0o600
        or value.st_uid != OWNER_UID
        or value.st_gid != OWNER_GID
        or value.st_nlink != 1
        or (device is not None and value.st_dev != device)
        or (inode is not None and value.st_ino != inode)
    ):
        raise LockHelperError("device lock topology differs")
    return value


def create() -> dict[str, object]:
    if RUNTIME_DIR.exists() or RUNTIME_DIR.is_symlink() or LOCK_PATH.exists() or LOCK_PATH.is_symlink():
        raise LockHelperError("runtime directory or device lock already exists")
    created_directory = False
    created_lock = False
    try:
        os.mkdir(RUNTIME_DIR, 0o750)
        created_directory = True
        os.chown(RUNTIME_DIR, OWNER_UID, OWNER_GID, follow_symlinks=False)
        os.chmod(RUNTIME_DIR, 0o750, follow_symlinks=False)
        directory = RUNTIME_DIR.stat(follow_symlinks=False)
        if (
            RUNTIME_DIR.is_symlink()
            or not stat.S_ISDIR(directory.st_mode)
            or stat.S_IMODE(directory.st_mode) != 0o750
            or directory.st_uid != OWNER_UID
            or directory.st_gid != OWNER_GID
        ):
            raise LockHelperError("runtime directory topology differs")
        descriptor = os.open(LOCK_PATH, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            created_lock = True
            os.fchown(descriptor, OWNER_UID, OWNER_GID)
            os.fchmod(descriptor, 0o600)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _validate_regular(LOCK_PATH)
        return {
            "status": "created",
            "runtime_directory_created": True,
            "runtime_directory": _metadata(RUNTIME_DIR),
            "lock": _metadata(LOCK_PATH),
        }
    except BaseException:
        if created_lock:
            try:
                os.unlink(LOCK_PATH)
            except OSError:
                pass
        if created_directory:
            try:
                os.rmdir(RUNTIME_DIR)
            except OSError:
                pass
        raise


def remove(device: int, inode: int) -> dict[str, object]:
    _validate_regular(LOCK_PATH, device=device, inode=inode)
    directory = RUNTIME_DIR.stat(follow_symlinks=False)
    if (
        RUNTIME_DIR.is_symlink()
        or not stat.S_ISDIR(directory.st_mode)
        or stat.S_IMODE(directory.st_mode) != 0o750
        or directory.st_uid != OWNER_UID
        or directory.st_gid != OWNER_GID
    ):
        raise LockHelperError("runtime directory topology differs")
    os.unlink(LOCK_PATH)
    if any(RUNTIME_DIR.iterdir()):
        raise LockHelperError("wrapper-created runtime directory is not empty")
    os.rmdir(RUNTIME_DIR)
    return {"status": "removed", "device": device, "inode": inode, "runtime_directory_removed": True}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    subparsers.add_parser("create")
    remove_parser = subparsers.add_parser("remove")
    remove_parser.add_argument("--device", type=int, required=True)
    remove_parser.add_argument("--inode", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        if os.geteuid() != 0:
            raise LockHelperError("lock helper requires effective UID 0")
        result = create() if args.action == "create" else remove(args.device, args.inode)
        print(json.dumps(result, sort_keys=True))
        return 0
    except (LockHelperError, OSError, ValueError) as error:
        print(f"SQ8 device lock helper failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
