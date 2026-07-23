from __future__ import annotations

import importlib.util
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL_PATH = ROOT / "tools/build-sq8-worker-release.py"
PROMOTION_PATH = ROOT / "tools/sq8_serving_promotion.py"
SPEC = importlib.util.spec_from_file_location(
    "test_build_sq8_worker_release_tool", TOOL_PATH
)
assert SPEC is not None and SPEC.loader is not None
TOOL = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TOOL
SPEC.loader.exec_module(TOOL)
PROMOTION_SPEC = importlib.util.spec_from_file_location(
    "test_build_sq8_worker_release_promotion",
    PROMOTION_PATH,
)
assert PROMOTION_SPEC is not None and PROMOTION_SPEC.loader is not None
PROMOTION = importlib.util.module_from_spec(PROMOTION_SPEC)
sys.modules[PROMOTION_SPEC.name] = PROMOTION
PROMOTION_SPEC.loader.exec_module(PROMOTION)


def source_tree(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    root.mkdir()
    for relative in TOOL.SOURCE_INPUTS:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture:{relative}\n", encoding="utf-8")
    return root


class FakeRunner:
    def __init__(
        self,
        root: Path,
        *,
        dirty: bool = False,
        attached: bool = False,
        build_bytes: bytes = b"sealed-sq8-worker\n",
    ) -> None:
        self.root = root
        self.dirty = dirty
        self.attached = attached
        self.build_bytes = build_bytes
        self.calls: list[tuple[list[str], Path]] = []

    def __call__(self, argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        cwd = Path(kwargs["cwd"])
        self.calls.append((list(argv), cwd))
        executable = Path(argv[0]).name
        if executable == "git":
            arguments = argv[1:]
            if arguments == ["rev-parse", "--show-toplevel"]:
                return subprocess.CompletedProcess(argv, 0, str(self.root) + "\n", "")
            if arguments == ["rev-parse", "HEAD"]:
                return subprocess.CompletedProcess(argv, 0, "a" * 40 + "\n", "")
            if arguments == ["rev-parse", "HEAD^{tree}"]:
                return subprocess.CompletedProcess(argv, 0, "b" * 40 + "\n", "")
            if arguments == ["status", "--porcelain=v1", "--untracked-files=all"]:
                output = " M dirty\n" if self.dirty else ""
                return subprocess.CompletedProcess(argv, 0, output, "")
            if arguments == ["symbolic-ref", "-q", "HEAD"]:
                if self.attached:
                    return subprocess.CompletedProcess(
                        argv, 0, "refs/heads/main\n", ""
                    )
                return subprocess.CompletedProcess(argv, 1, "", "")
            if arguments == ["show", "-s", "--format=%ct", "HEAD"]:
                return subprocess.CompletedProcess(argv, 0, "1784865600\n", "")
            raise AssertionError(arguments)
        if executable == "cargo" and len(argv) > 1 and argv[1] == "build":
            environment = kwargs["env"]
            target = Path(environment["CARGO_TARGET_DIR"])
            worker = target / "release/ullm-sq8-worker"
            worker.parent.mkdir(parents=True)
            worker.write_bytes(self.build_bytes)
            worker.chmod(0o755)
            return subprocess.CompletedProcess(argv, 0, "", "compiled")
        versions = {
            "cargo": "cargo 1.96.0",
            "rustc": "rustc 1.96.0",
            "c++": "c++ 13.3.0",
            "hipcc": "HIP 7.2.0",
        }
        if executable in versions:
            return subprocess.CompletedProcess(argv, 0, versions[executable] + "\n", "")
        raise AssertionError(argv)


def test_build_release_seals_clean_detached_source_and_worker(
    tmp_path: Path,
) -> None:
    repo = source_tree(tmp_path)
    output = tmp_path / "release"
    target = tmp_path / "target"
    runner = FakeRunner(repo)

    receipt = TOOL.build_release(repo, output, target, runner=runner)

    assert receipt["schema_version"] == TOOL.RECEIPT_SCHEMA
    assert receipt["source"]["commit"] == "a" * 40
    assert receipt["source"]["tree"] == "b" * 40
    assert receipt["source"]["detached"] is True
    assert receipt["worker"]["sha256"] == TOOL.sha256_file(
        output / "ullm-sq8-worker"
    )
    assert receipt["worker"]["path"] == os.fspath(output / "ullm-sq8-worker")
    assert receipt["source"]["repository_root"] == os.fspath(repo)
    assert receipt["source"]["worktree_clean"] is True
    assert receipt["build"]["result"] == "success"
    assert [entry["path"] for entry in receipt["inputs"]] == sorted(
        TOOL.SOURCE_INPUTS,
        key=lambda value: value.encode("utf-8"),
    )
    assert stat.S_IMODE(output.stat().st_mode) == 0o555
    assert stat.S_IMODE((output / "ullm-sq8-worker").stat().st_mode) == 0o555
    for name in (
        "README.md",
        "build-provenance.json",
        "build-receipt.json",
        "SHA256SUMS",
        "SEALED.json",
    ):
        metadata = (output / name).stat()
        assert stat.S_IMODE(metadata.st_mode) == 0o444
        assert metadata.st_nlink == 1
    assert (
        PROMOTION.validate_build_receipt(
            output / "build-receipt.json",
            verify_live_source=False,
        )
        == receipt
    )
    seal = json.loads((output / "SEALED.json").read_text(encoding="ascii"))
    assert seal["complete"] is True
    assert seal["worker_sha256"] == receipt["worker"]["sha256"]
    assert seal["build_provenance_sha256"] == TOOL.sha256_file(
        output / "build-provenance.json"
    )
    sums = dict(
        line.split("  ", 1)
        for line in (output / "SHA256SUMS").read_text(encoding="ascii").splitlines()
    )
    for digest, name in sums.items():
        assert digest == TOOL.sha256_file(output / name)


@pytest.mark.parametrize(
    ("dirty", "attached", "match"),
    [
        (True, False, "not clean"),
        (False, True, "not detached"),
    ],
)
def test_build_rejects_nonrelease_source_state(
    tmp_path: Path, dirty: bool, attached: bool, match: str
) -> None:
    repo = source_tree(tmp_path)
    runner = FakeRunner(repo, dirty=dirty, attached=attached)
    with pytest.raises(TOOL.BuildError, match=match):
        TOOL.build_release(
            repo,
            tmp_path / "release",
            tmp_path / "target",
            runner=runner,
        )
    assert not (tmp_path / "release").exists()


def test_build_rejects_existing_output_or_target_without_clobber(
    tmp_path: Path,
) -> None:
    repo = source_tree(tmp_path)
    runner = FakeRunner(repo)
    output = tmp_path / "release"
    output.mkdir()
    marker = output / "keep"
    marker.write_text("keep", encoding="ascii")
    with pytest.raises(TOOL.BuildError, match="already exists"):
        TOOL.build_release(
            repo,
            output,
            tmp_path / "target",
            runner=runner,
        )
    assert marker.read_text(encoding="ascii") == "keep"
    assert runner.calls == []

    marker.unlink()
    output.rmdir()
    target = tmp_path / "target"
    target.mkdir()
    with pytest.raises(TOOL.BuildError, match="already exists"):
        TOOL.build_release(repo, output, target, runner=runner)
    assert not output.exists()


def test_build_rejects_ambient_compile_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = source_tree(tmp_path)
    runner = FakeRunner(repo)
    monkeypatch.setenv("RUSTFLAGS", "-C target-cpu=native")
    with pytest.raises(TOOL.BuildError, match="ambient build override"):
        TOOL.build_release(
            repo,
            tmp_path / "release",
            tmp_path / "target",
            runner=runner,
        )
    assert not (tmp_path / "release").exists()


def test_second_release_cannot_replace_first(tmp_path: Path) -> None:
    repo = source_tree(tmp_path)
    output = tmp_path / "release"
    first = FakeRunner(repo, build_bytes=b"first\n")
    TOOL.build_release(repo, output, tmp_path / "target-1", runner=first)
    original = (output / "ullm-sq8-worker").read_bytes()
    with pytest.raises(TOOL.BuildError, match="already exists"):
        TOOL.build_release(
            repo,
            output,
            tmp_path / "target-2",
            runner=FakeRunner(repo, build_bytes=b"second\n"),
        )
    assert (output / "ullm-sq8-worker").read_bytes() == original
