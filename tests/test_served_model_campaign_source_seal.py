from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tools" / "served_model_campaign_source_seal.py"
SPEC = importlib.util.spec_from_file_location(
    "test_served_model_campaign_source_seal_module",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
SEAL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SEAL
SPEC.loader.exec_module(SEAL)


def _mkdir(path: Path) -> None:
    path.mkdir()
    path.chmod(0o755)


def sealed_repository(parent: Path, name: str = "source") -> Path:
    root = parent / name
    _mkdir(root)
    git = root / ".git"
    _mkdir(git)
    objects = git / "objects"
    _mkdir(objects)
    info = objects / "info"
    _mkdir(info)
    config = git / "config"
    config.write_text(
        "[core]\n\trepositoryformatversion = 0\n\tfsmonitor = attacker\n",
        encoding="ascii",
    )
    config.chmod(0o644)
    tools = root / "tools"
    _mkdir(tools)
    script = tools / "campaign.py"
    script.write_text("raise SystemExit(0)\n", encoding="ascii")
    script.chmod(0o644)
    return root


def test_safe_repository_below_root_owned_sticky_tmp_is_sealed(
    tmp_path: Path,
) -> None:
    root = sealed_repository(tmp_path)
    sealed = SEAL.capture_source_seal(root, required_uid=os.geteuid())

    assert sealed.root == root
    assert sealed.required_uid == os.geteuid()
    assert len(sealed.fingerprint_sha256) == 64
    assert {entry.relative_path for entry in sealed.entries} >= {
        ".",
        ".git",
        ".git/config",
        "tools/campaign.py",
    }
    assert SEAL.require_source_seal(
        sealed,
        required_uid=os.geteuid(),
    ) == sealed


def test_group_world_writable_ancestor_and_entry_are_rejected(
    tmp_path: Path,
) -> None:
    shared = tmp_path / "shared"
    _mkdir(shared)
    root = sealed_repository(shared)

    shared.chmod(0o775)
    with pytest.raises(SEAL.SourceSealError, match="ancestry.*writable"):
        SEAL.capture_source_seal(root, required_uid=os.geteuid())

    shared.chmod(0o755)
    script = root / "tools" / "campaign.py"
    script.chmod(0o664)
    with pytest.raises(SEAL.SourceSealError, match="entry.*writable"):
        SEAL.capture_source_seal(root, required_uid=os.geteuid())


def test_symlink_ancestry_and_entry_are_rejected(tmp_path: Path) -> None:
    real_parent = tmp_path / "real"
    _mkdir(real_parent)
    root = sealed_repository(real_parent)
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)

    with pytest.raises(SEAL.SourceSealError, match="symlink-free"):
        SEAL.capture_source_seal(
            linked_parent / root.name,
            required_uid=os.geteuid(),
        )

    link = root / "tools" / "linked.py"
    link.symlink_to(root / "tools" / "campaign.py")
    with pytest.raises(SEAL.SourceSealError, match="symbolic link"):
        SEAL.capture_source_seal(root, required_uid=os.geteuid())


def test_linked_worktree_git_file_and_external_gitdir_are_rejected(
    tmp_path: Path,
) -> None:
    root = tmp_path / "linked-worktree"
    _mkdir(root)
    external = tmp_path / "external.git"
    _mkdir(external)
    gitfile = root / ".git"
    gitfile.write_text(f"gitdir: {external}\n", encoding="ascii")
    gitfile.chmod(0o644)

    with pytest.raises(SEAL.SourceSealError, match="internal directory"):
        SEAL.capture_source_seal(root, required_uid=os.geteuid())


def test_regular_hardlink_and_git_alternates_are_rejected(
    tmp_path: Path,
) -> None:
    root = sealed_repository(tmp_path)
    source = root / "tools" / "campaign.py"
    os.link(source, root / "tools" / "campaign-copy.py")
    with pytest.raises(SEAL.SourceSealError, match="hard-linked"):
        SEAL.capture_source_seal(root, required_uid=os.geteuid())

    (root / "tools" / "campaign-copy.py").unlink()
    alternates = root / ".git" / "objects" / "info" / "alternates"
    alternates.write_text("/attacker/objects\n", encoding="ascii")
    alternates.chmod(0o644)
    with pytest.raises(SEAL.SourceSealError, match="alternates"):
        SEAL.capture_source_seal(root, required_uid=os.geteuid())


def test_special_file_is_rejected(tmp_path: Path) -> None:
    root = sealed_repository(tmp_path)
    fifo = root / "tools" / "campaign.fifo"
    os.mkfifo(fifo, 0o600)
    with pytest.raises(SEAL.SourceSealError, match="special file"):
        SEAL.capture_source_seal(root, required_uid=os.geteuid())


def test_posix_acl_on_entry_and_ancestor_is_rejected(tmp_path: Path) -> None:
    if not Path("/usr/bin/setfacl").exists():
        pytest.skip("setfacl is unavailable")
    root = sealed_repository(tmp_path)
    script = root / "tools" / "campaign.py"
    completed = subprocess.run(
        ["/usr/bin/setfacl", "-m", "u:nobody:r--", str(script)],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if completed.returncode != 0:
        pytest.skip("test filesystem does not support POSIX ACLs")
    with pytest.raises(SEAL.SourceSealError, match="POSIX ACL"):
        SEAL.capture_source_seal(root, required_uid=os.geteuid())

    subprocess.run(
        ["/usr/bin/setfacl", "-b", str(script)],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    completed = subprocess.run(
        ["/usr/bin/setfacl", "-m", "u:nobody:r-x", str(root.parent)],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if completed.returncode != 0:
        pytest.skip("test filesystem does not support directory ACLs")
    with pytest.raises(SEAL.SourceSealError, match="ancestry.*POSIX ACL"):
        SEAL.capture_source_seal(root, required_uid=os.geteuid())


def test_fingerprint_detects_content_replacement_and_restoration(
    tmp_path: Path,
) -> None:
    root = sealed_repository(tmp_path)
    script = root / "tools" / "campaign.py"
    original = script.read_bytes()
    sealed = SEAL.capture_source_seal(root, required_uid=os.geteuid())

    replacement = root / "tools" / "replacement.py"
    replacement.write_bytes(b"raise SystemExit(99)\n")
    replacement.chmod(0o644)
    script.rename(root / "tools" / "original.py")
    replacement.rename(script)
    script.unlink()
    (root / "tools" / "original.py").rename(script)
    assert script.read_bytes() == original

    with pytest.raises(SEAL.SourceSealError, match="seal changed"):
        SEAL.require_source_seal(sealed, required_uid=os.geteuid())


def test_git_invocation_is_fixed_path_and_environment_is_allowlisted() -> None:
    argv = SEAL.git_argv(["status", "--porcelain=v1"])
    assert argv[0] == "/usr/bin/git"
    assert argv[: len(SEAL.GIT_COMMAND_PREFIX)] == list(SEAL.GIT_COMMAND_PREFIX)
    assert argv.index("core.fsmonitor=false") < argv.index("status")
    assert argv.index("core.hooksPath=/dev/null") < argv.index("status")
    assert argv.index("core.useReplaceRefs=false") < argv.index("status")

    environment = SEAL.git_environment()
    assert environment == SEAL.GIT_ENVIRONMENT
    assert environment["GIT_OPTIONAL_LOCKS"] == "0"
    assert environment["GIT_CONFIG_GLOBAL"] == "/dev/null"
    assert environment["GIT_CONFIG_NOSYSTEM"] == "1"
    assert "LD_PRELOAD" not in environment
    assert "PYTHONPATH" not in environment


def test_hardened_git_status_does_not_execute_repository_fsmonitor(
    tmp_path: Path,
) -> None:
    root = tmp_path / "source"
    root.mkdir()
    script = root / "campaign.py"
    script.write_text("raise SystemExit(0)\n", encoding="ascii")
    subprocess.run(
        ["/usr/bin/git", "init", "--quiet", str(root)],
        check=True,
    )
    subprocess.run(
        ["/usr/bin/git", "-C", str(root), "config", "user.email", "fixture@example"],
        check=True,
    )
    subprocess.run(
        ["/usr/bin/git", "-C", str(root), "config", "user.name", "Fixture"],
        check=True,
    )
    subprocess.run(
        ["/usr/bin/git", "-C", str(root), "add", "campaign.py"],
        check=True,
    )
    subprocess.run(
        ["/usr/bin/git", "-C", str(root), "commit", "--quiet", "-m", "fixture"],
        check=True,
    )
    marker = tmp_path / "fsmonitor-executed"
    hook = tmp_path / "malicious-fsmonitor"
    hook.write_text(
        f"#!/bin/sh\n/usr/bin/touch {marker}\n",
        encoding="ascii",
    )
    hook.chmod(0o755)
    subprocess.run(
        [
            "/usr/bin/git",
            "-C",
            str(root),
            "config",
            "core.fsmonitor",
            str(hook),
        ],
        check=True,
    )
    for entry in (root, *root.rglob("*")):
        entry.chmod(0o755 if entry.is_dir() else 0o644)

    sealed = SEAL.capture_source_seal(root, required_uid=os.geteuid())
    completed = subprocess.run(
        SEAL.git_argv(
            [
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
                "--ignore-submodules=all",
                "--no-renames",
            ]
        ),
        cwd=root,
        env=SEAL.git_environment(),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert completed.returncode == 0
    assert completed.stdout == ""
    assert not marker.exists()
    assert SEAL.require_source_seal(
        sealed,
        required_uid=os.geteuid(),
    ) == sealed


@pytest.mark.parametrize(
    "value",
    (
        Path("relative/source"),
        Path("/tmp/../tmp/source"),
        Path("//tmp/source"),
    ),
)
def test_source_root_must_be_lexical_absolute(value: Path) -> None:
    with pytest.raises(SEAL.SourceSealError, match="lexical absolute"):
        SEAL.capture_source_seal(value, required_uid=os.geteuid())
