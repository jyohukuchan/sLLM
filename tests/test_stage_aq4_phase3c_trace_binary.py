from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "stage-aq4-phase3c-trace-binary.py"
BINARY = "ullm-aq4-differential-trace"
COMMIT = "5a0fb4c50476d5153ced22bd6847c2729bfdb975"


def run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(TOOL), *arguments],
        text=True,
        capture_output=True,
        check=False,
    )


def create_hardlinked_cargo_output(root: Path) -> Path:
    source = root / BINARY
    source.write_bytes(b"detached Phase 3c trace binary\n" * 1024)
    source.chmod(0o700)
    os.link(source, root / "deps-peer")
    assert source.stat().st_nlink == 2
    return source


def test_create_new_staging_copy_breaks_cargo_hardlink_and_records_sha(tmp_path: Path) -> None:
    source = create_hardlinked_cargo_output(tmp_path)
    stage = tmp_path / "trace-binary-staging"

    result = run(
        "--source",
        str(source),
        "--output",
        str(stage),
        "--trace-tooling-commit",
        COMMIT,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    staged = stage / BINARY
    expected_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    assert source.stat().st_nlink == 2
    assert staged.is_file() and not staged.is_symlink()
    assert staged.stat().st_nlink == 1
    assert stat.S_IMODE(staged.stat().st_mode) == 0o555
    assert hashlib.sha256(staged.read_bytes()).hexdigest() == expected_sha
    assert stat.S_IMODE(stage.stat().st_mode) == 0o555
    assert report["status"] == "valid"
    assert report["binary"]["sha256"] == expected_sha
    receipt = json.loads((stage / "staging-receipt.json").read_text(encoding="utf-8"))
    assert receipt["source"]["sha256"] == expected_sha
    assert receipt["source"]["nlink"] == 2
    assert receipt["staged_binary"]["sha256"] == expected_sha
    assert receipt["staged_binary"]["nlink"] == 1
    assert (stage / "SHA256SUMS").read_text(encoding="utf-8") == (
        f"{expected_sha}  {BINARY}\n"
        f"{hashlib.sha256((stage / 'staging-receipt.json').read_bytes()).hexdigest()}  staging-receipt.json\n"
    )

    verified = run(
        "--verify",
        "--source",
        str(source),
        "--output",
        str(stage),
        "--trace-tooling-commit",
        COMMIT,
    )
    assert verified.returncode == 0, verified.stderr
    assert json.loads(verified.stdout)["binary"]["nlink"] == 1


def test_existing_or_hardlinked_staging_is_rejected(tmp_path: Path) -> None:
    source = create_hardlinked_cargo_output(tmp_path)
    stage = tmp_path / "trace-binary-staging"
    created = run(
        "--source",
        str(source),
        "--output",
        str(stage),
        "--trace-tooling-commit",
        COMMIT,
    )
    assert created.returncode == 0, created.stderr

    rerun = run(
        "--source",
        str(source),
        "--output",
        str(stage),
        "--trace-tooling-commit",
        COMMIT,
    )
    assert rerun.returncode != 0
    assert "refusing to overwrite existing trace-binary staging directory" in rerun.stderr

    stage.chmod(0o755)
    os.link(stage / BINARY, stage / "hardlink-peer")
    stage.chmod(0o555)
    invalid = run(
        "--verify",
        "--source",
        str(source),
        "--output",
        str(stage),
        "--trace-tooling-commit",
        COMMIT,
    )
    assert invalid.returncode != 0
    assert "members differ" in invalid.stderr or "nlink=1" in invalid.stderr
