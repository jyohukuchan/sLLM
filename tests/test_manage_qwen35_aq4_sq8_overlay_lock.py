from __future__ import annotations

import importlib.util
import os
import stat
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/manage-qwen35-aq4-sq8-overlay-lock.py"
SPEC = importlib.util.spec_from_file_location("sq8_lock_helper", TOOL)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


@pytest.fixture
def paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> tuple[Path, Path]:
    runtime = tmp_path / "ullm"
    lock = runtime / "device-1.lock"
    monkeypatch.setattr(MODULE, "RUNTIME_DIR", runtime)
    monkeypatch.setattr(MODULE, "LOCK_PATH", lock)
    monkeypatch.setattr(MODULE, "OWNER_UID", os.getuid())
    monkeypatch.setattr(MODULE, "OWNER_GID", os.getgid())
    return runtime, lock


def test_create_and_remove_exact_topology(paths: tuple[Path, Path]) -> None:
    runtime, lock = paths
    created = MODULE.create()
    lock_stat = lock.stat(follow_symlinks=False)
    assert created["status"] == "created"
    assert stat.S_IMODE(runtime.stat().st_mode) == 0o750
    assert stat.S_IMODE(lock_stat.st_mode) == 0o600 and lock_stat.st_nlink == 1

    removed = MODULE.remove(lock_stat.st_dev, lock_stat.st_ino)

    assert removed["runtime_directory_removed"] is True
    assert not runtime.exists()


@pytest.mark.parametrize("kind", ["directory", "symlink"])
def test_create_rejects_preexisting_or_symlink(
    paths: tuple[Path, Path], tmp_path: Path, kind: str
) -> None:
    runtime, _lock = paths
    if kind == "directory":
        runtime.mkdir()
    else:
        target = tmp_path / "target"
        target.mkdir()
        runtime.symlink_to(target, target_is_directory=True)
    with pytest.raises(MODULE.LockHelperError, match="already exists"):
        MODULE.create()


def test_remove_rejects_wrong_mode_owner_and_inode(
    paths: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    runtime, lock = paths
    created = MODULE.create()
    device = created["lock"]["device"]
    inode = created["lock"]["inode"]
    lock.chmod(0o644)
    with pytest.raises(MODULE.LockHelperError, match="topology"):
        MODULE.remove(device, inode)
    lock.chmod(0o600)
    monkeypatch.setattr(MODULE, "OWNER_UID", os.getuid() + 1)
    with pytest.raises(MODULE.LockHelperError, match="topology"):
        MODULE.remove(device, inode)
    monkeypatch.setattr(MODULE, "OWNER_UID", os.getuid())
    with pytest.raises(MODULE.LockHelperError, match="topology"):
        MODULE.remove(device, inode + 1)
    MODULE.remove(device, inode)
    assert not runtime.exists()


def test_remove_only_rmdirs_an_empty_runtime_directory(paths: tuple[Path, Path]) -> None:
    runtime, lock = paths
    created = MODULE.create()
    extra = runtime / "unexpected"
    extra.write_text("x", encoding="ascii")
    with pytest.raises(MODULE.LockHelperError, match="not empty"):
        MODULE.remove(created["lock"]["device"], created["lock"]["inode"])
    assert not lock.exists() and runtime.is_dir() and extra.is_file()
