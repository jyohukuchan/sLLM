#!/usr/bin/env python3
"""Probe the pre-existing Phase 3c lock without creating or modifying it."""

from __future__ import annotations

import errno
import fcntl
import json
import os
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path


def main(output_path: Path, lock_path: Path) -> int:
    payload: dict[str, object] = {
        "schema_version": "ullm.aq4_phase3c_service_window_lock_after_stop.v1",
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "path": str(lock_path),
        "probe": {
            "create_flag_used": False,
            "open_flags": ["O_RDWR", "O_NOFOLLOW", "O_CLOEXEC"],
            "flock": "LOCK_EX|LOCK_NB",
        },
    }
    flags = os.O_RDWR | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW

    try:
        before = os.lstat(lock_path)
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            payload.update({"acquirable": False, "reason": "not_regular_file"})
            status = 40
        else:
            descriptor = os.open(lock_path, flags)
            try:
                opened = os.fstat(descriptor)
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                payload.update(
                    {
                        "acquirable": True,
                        "reason": "acquired_and_released",
                        "lstat": {
                            "device": before.st_dev,
                            "inode": before.st_ino,
                            "mode_octal": format(stat.S_IMODE(before.st_mode), "04o"),
                        },
                        "opened": {"device": opened.st_dev, "inode": opened.st_ino},
                    }
                )
                status = 0
            finally:
                os.close(descriptor)
    except OSError as error:
        payload.update(
            {
                "acquirable": False,
                "reason": "os_error",
                "errno": error.errno,
                "errno_name": errno.errorcode.get(error.errno, "UNKNOWN"),
                "error": str(error),
            }
        )
        status = 40

    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return status


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: probe-aq4-phase3c-existing-lock.py OUTPUT LOCK")
    raise SystemExit(main(Path(sys.argv[1]), Path(sys.argv[2])))
