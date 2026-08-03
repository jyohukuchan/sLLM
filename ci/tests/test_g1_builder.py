"""H0 host-only tests for the G1 builder and artifact contract.

The fake runner never starts a GPU process; a passing test is not G1 GPU
evidence.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ci/tools"))

import build_g1_runtime as builder  # noqa: E402
from validate_g1_contracts import (  # noqa: E402
    inspect_g1_runtime_artifact,
    parse_g1_device_readobj,
    validate_artifact_metadata,
    validate_g1_matrix,
)


SHA = "a" * 40
TREE = "b" * 40
KERNEL_SYMBOL = "_ZN12_GLOBAL__N_118evidence_transformEPKhPhm"


def host_readobj_fixture() -> str:
    return """File: final
Format: elf64-x86-64
Arch: x86_64
Sections [
  Section {
    Name: .text (1)
    Size: 64
  }
  Section {
    Name: .hip_fatbin (2)
    Size: 128
  }
]
"""


def device_readobj_fixture(target: str, *, wavefront_size: int = 32, flags: str | None = None) -> str:
    flags = flags or {"gfx1030": "36", "gfx1201": "4e"}[target]
    return f"""File: device-code-object.elf
Format: elf64-amdgpu
Arch: amdgcn
ElfHeader {{
  Ident {{
    ABIVersion: 4
  }}
  Flags [ (0x{flags})
  ]
}}
Sections [
  Section {{
    Name: .text (1)
    Size: 64
  }}
]
Symbols [
  Symbol {{
    Name:  (0)
    Type: None (0x0)
    Section: Undefined (0x0)
  }}
  Symbol {{
    Name: {KERNEL_SYMBOL} (1)
    Type: Function (0x2)
    Section: .text (0x1)
  }}
  Symbol {{
    Name: {KERNEL_SYMBOL}.kd (2)
    Type: Object (0x1)
    Section: .rodata (0x2)
  }}
]
NoteSections [
  NoteSection {{
    .name: {KERNEL_SYMBOL}
    .symbol: {KERNEL_SYMBOL}.kd
    .wavefront_size: {wavefront_size}
  }}
]
amdhsa.target:   amdgcn-amd-amdhsa--{target}
"""


class FakeRunner:
    """Mock the compiler/toolchain boundary without starting a GPU process."""

    def __init__(
        self,
        *,
        binary: bytes | None = None,
        target: str = "gfx1030",
        bundles: list[str] | None = None,
        host_output: str | None = None,
        device_output: str | None = None,
    ) -> None:
        self.calls: list[dict[str, object]] = []
        self.target = target
        self.bundles = bundles or [
            f"hipv4-amdgcn-amd-amdhsa--{target}",
            "host-x86_64-unknown-linux-gnu-",
        ]
        self.host_output = host_output or host_readobj_fixture()
        self.device_output = device_output or device_readobj_fixture(target)
        self.binary = binary or (
            b"not-an-ELF-but-the-pinned-tool-fixture-is-authoritative\n"
            b"libamdhip64.so\0sllm_hip_evidence_submit\0"
        )

    def __call__(self, argv, *, cwd=None, env=None, timeout=None):
        command = tuple(str(item) for item in argv)
        self.calls.append({"argv": command, "cwd": cwd, "env": env, "timeout": timeout})
        if command[:3] == ("git", "rev-parse", "--verify"):
            output = SHA if command[-1] == "HEAD^{commit}" else TREE
            return builder.CommandOutput(command, 0, (output + "\n").encode(), b"")
        if command[:2] == ("git", "status"):
            return builder.CommandOutput(command, 0, b"", b"")
        if command[-1] == "--version":
            return builder.CommandOutput(command, 0, b"AMD clang version 23.0.0git\n", b"")
        if command[:3] == ("cargo", "+1.97.1", "build"):
            assert env is not None
            target_dir = Path(env["CARGO_TARGET_DIR"])
            binary = target_dir / "release" / builder.BINARY_NAME
            binary.parent.mkdir(parents=True, exist_ok=True)
            binary.write_bytes(self.binary)
            binary.chmod(0o700)
            return builder.CommandOutput(command, 0, b"", b"")
        if command[0] == "/opt/rocm/lib/llvm/bin/llvm-objcopy":
            destination = next(item.split("=", 2)[2] for item in command if item.startswith("--dump-section=.hip_fatbin="))
            Path(destination).write_bytes(b"deterministic-fatbin")
            return builder.CommandOutput(command, 0, b"", b"")
        if command[0] == "/opt/rocm/lib/llvm/bin/clang-offload-bundler" and "--list" in command:
            return builder.CommandOutput(command, 0, ("\n".join(self.bundles) + "\n").encode(), b"")
        if command[0] == "/opt/rocm/lib/llvm/bin/clang-offload-bundler" and "--unbundle" in command:
            destination = next(item.split("=", 1)[1] for item in command if item.startswith("--output="))
            Path(destination).write_bytes(b"deterministic-device-code-object")
            return builder.CommandOutput(command, 0, b"", b"")
        if command[0] == "/opt/rocm/lib/llvm/bin/llvm-readobj":
            output = self.device_output if command[-1].endswith("device-code-object.elf") else self.host_output
            return builder.CommandOutput(command, 0, output.encode(), b"")
        raise AssertionError(f"unexpected subprocess argv: {command}")


class G1BuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.roots: list[Path] = []

    def tearDown(self) -> None:
        for path in self.roots:
            shutil.rmtree(path, ignore_errors=True)

    def private_root(self, name: str = "sllm-g1-test") -> Path:
        root = Path(tempfile.mkdtemp(prefix=name + "-", dir="/tmp"))
        root.chmod(0o700)
        self.roots.append(root)
        return root

    def build(self, fake: FakeRunner | None = None, **kwargs):
        root = kwargs.pop("output_root", self.private_root())
        row = kwargs.pop("row_id", "g1-gfx1030")
        fake = fake or FakeRunner(target=row.removeprefix("g1-"))
        with patch.dict(builder.os.environ, {}, clear=True), patch.object(
            builder, "verify_candidate"
        ) as candidate, patch.object(builder, "validate_toolchain") as toolchain:
            result = builder.build_runtime_artifact(
                repo=ROOT,
                row_id=row,
                reviewed_sha=kwargs.pop("reviewed_sha", SHA),
                tested_sha=kwargs.pop("tested_sha", SHA),
                workflow_sha=kwargs.pop("workflow_sha", SHA),
                tree_oid=kwargs.pop("tree_oid", TREE),
                output_dir=kwargs.pop("output_dir", root / row),
                runner=fake,
                **kwargs,
            )
        candidate.assert_called()
        toolchain.assert_called_once()
        return result, fake

    def test_build_stages_only_dedicated_binary_metadata_and_strict_sidecars(self) -> None:
        result, fake = self.build()
        self.assertEqual(result.row_id, "g1-gfx1030")
        self.assertEqual(
            {path.name for path in result.output_dir.iterdir()},
            {
                builder.BINARY_NAME,
                builder.BINARY_NAME + ".sha256",
                builder.METADATA_NAME,
                builder.METADATA_NAME + ".sha256",
            },
        )
        metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["candidate"], {
            "reviewed_sha": SHA,
            "tested_sha": SHA,
            "workflow_sha": SHA,
            "git_tree_oid": TREE,
            "worktree_clean": True,
            "revision_input": "full-sha",
        })
        self.assertEqual(metadata["gpu"], {
            "bdf": "0000:03:00.0",
            "uuid": "GPU-76a08c022586fed6",
            "target": "gfx1030",
        })
        self.assertEqual(metadata["scope"]["model_used"], False)
        self.assertEqual(metadata["scope"]["cpu_fallback_allowed"], False)
        self.assertEqual(metadata["scope"]["cpu_fallback_used"], False)
        validate_g1_matrix(ROOT)
        validate_artifact_metadata(
            metadata,
            result.artifact_path,
            result.metadata_path,
            expected=next(row for row in validate_g1_matrix(ROOT)["rows"] if row["row_id"] == result.row_id),
            identity={
                "run_id": "g1-builder",
                "run_attempt": 1,
                "reviewed_sha": SHA,
                "tested_sha": SHA,
                "workflow_sha": SHA,
                "git_tree_oid": TREE,
            },
            repo=ROOT,
            tool_runner=fake,
        )
        cargo = next(call for call in fake.calls if tuple(call["argv"])[0] == "cargo")
        command = tuple(cargo["argv"])
        self.assertEqual(command, (
            "cargo", "+1.97.1", "build", "--locked", "--offline", "--release",
            "--package", "sllm-hip", "--bin", "sllm-hip-evidence",
        ))
        self.assertNotIn("--all-features", command)
        self.assertNotIn("--features", command)
        environment = cargo["env"]
        self.assertEqual(environment["SLLM_ENABLE_HIP_RUNTIME"], "1")
        self.assertEqual(environment["SLLM_ENABLE_HIP_COMPILE_PROBE"], "0")
        self.assertEqual(environment["CMAKE_HIP_ARCHITECTURES"], "gfx1030")
        self.assertEqual(environment["ROCM_PATH"], "/opt/rocm")
        self.assertEqual(environment["SLLM_HIP_COMPILER"], "/opt/rocm/bin/amdclang++")
        self.assertTrue(Path(environment["CARGO_TARGET_DIR"]).is_relative_to(result.output_dir.parent))
        self.assertTrue(all(call["timeout"] is not None for call in fake.calls))

    def test_second_exact_row_binds_gfx1201_bdf_and_uuid(self) -> None:
        result, fake = self.build(row_id="g1-gfx1201")
        metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
        self.assertEqual(metadata["row_id"], "g1-gfx1201")
        self.assertEqual(metadata["target"], "gfx1201")
        self.assertEqual(metadata["gpu"], {
            "bdf": "0000:47:00.0",
            "uuid": "GPU-a8e9ddefa2d60f55",
            "target": "gfx1201",
        })
        cargo = next(call for call in fake.calls if tuple(call["argv"])[0] == "cargo")
        self.assertEqual(cargo["env"]["CMAKE_HIP_ARCHITECTURES"], "gfx1201")

    def test_candidate_must_be_complete_equal_current_clean_sha_and_tree(self) -> None:
        with self.assertRaises(builder.G1BuilderError):
            builder.build_runtime_artifact(
                repo=ROOT,
                row_id="g1-gfx1030",
                reviewed_sha="short",
                tested_sha=SHA,
                workflow_sha=SHA,
                tree_oid=TREE,
            )
        fake = FakeRunner()
        with self.assertRaises(builder.G1BuilderError):
            builder.verify_candidate(
                ROOT,
                {"reviewed_sha": SHA, "tested_sha": SHA, "workflow_sha": SHA, "git_tree_oid": TREE},
                runner=lambda *args, **kwargs: builder.CommandOutput(
                    tuple(args[0]), 0, b" M ci/tools/other.py\n", b""
                ),
            )
        with self.assertRaises(builder.G1BuilderError):
            builder.verify_candidate(
                ROOT,
                {"reviewed_sha": SHA, "tested_sha": SHA, "workflow_sha": SHA, "git_tree_oid": TREE},
                runner=lambda *args, **kwargs: builder.CommandOutput(
                    tuple(args[0]), 0,
                    ("c" * 40 + "\n").encode()
                    if tuple(args[0])[-1] == "HEAD^{commit}"
                    else (TREE + "\n").encode(),
                    b"",
                ),
            )

    def test_mismatched_candidate_sha_or_tree_is_rejected_before_cargo(self) -> None:
        fake = FakeRunner()
        with patch.object(builder, "verify_candidate") as verify:
            with self.assertRaises(builder.G1BuilderError):
                builder.build_runtime_artifact(
                    repo=ROOT,
                    row_id="g1-gfx1030",
                    reviewed_sha=SHA,
                    tested_sha="c" * 40,
                    workflow_sha=SHA,
                    tree_oid=TREE,
                    runner=fake,
                )
        verify.assert_not_called()

    def test_unknown_row_and_wrong_rocm_root_are_rejected(self) -> None:
        with self.assertRaises(builder.G1BuilderError):
            builder.build_runtime_artifact(
                repo=ROOT, row_id="g1-gfx1031", reviewed_sha=SHA,
                tested_sha=SHA, workflow_sha=SHA, tree_oid=TREE,
            )
        with self.assertRaises(builder.G1BuilderError):
            builder.build_runtime_artifact(
                repo=ROOT, row_id="g1-gfx1030", reviewed_sha=SHA,
                tested_sha=SHA, workflow_sha=SHA, tree_oid=TREE,
                rocm_root=Path("/opt/rocm-7.13"),
            )

    def test_repository_output_and_non_private_output_are_rejected(self) -> None:
        for output in (ROOT / "g1-gfx1030", Path("/tmp/not-g1-output/g1-gfx1030")):
            with self.subTest(output=output), patch.object(builder, "verify_candidate"), patch.object(
                builder, "validate_toolchain"
            ):
                with self.assertRaises(builder.G1BuilderError):
                    builder.build_runtime_artifact(
                        repo=ROOT, row_id="g1-gfx1030", reviewed_sha=SHA,
                        tested_sha=SHA, workflow_sha=SHA, tree_oid=TREE,
                        output_dir=output, runner=FakeRunner(),
                    )

    def test_symlinked_or_stale_output_is_rejected(self) -> None:
        root = self.private_root()
        outside = self.private_root("sllm-g1-outside")
        symlink = root / "g1-gfx1030"
        symlink.symlink_to(outside, target_is_directory=True)
        with patch.object(builder, "verify_candidate"), patch.object(builder, "validate_toolchain"):
            with self.assertRaises(builder.G1BuilderError):
                builder.build_runtime_artifact(
                    repo=ROOT, row_id="g1-gfx1030", reviewed_sha=SHA,
                    tested_sha=SHA, workflow_sha=SHA, tree_oid=TREE,
                    output_dir=symlink, runner=FakeRunner(),
                )
        stale_root = self.private_root()
        stale_row = stale_root / "g1-gfx1030"
        stale_row.mkdir(mode=0o700)
        with patch.object(builder, "verify_candidate"), patch.object(builder, "validate_toolchain"):
            with self.assertRaises(builder.G1BuilderError):
                builder.build_runtime_artifact(
                    repo=ROOT, row_id="g1-gfx1030", reviewed_sha=SHA,
                    tested_sha=SHA, workflow_sha=SHA, tree_oid=TREE,
                    output_dir=stale_row, runner=FakeRunner(),
                )

    def test_inherited_wrong_target_root_wrapper_or_rustflags_fails_closed(self) -> None:
        bad_values = (
            {"CMAKE_HIP_ARCHITECTURES": "gfx1201"},
            {"ROCM_PATH": "/opt/rocm-7.13"},
            {"SLLM_HIP_COMPILER": "/usr/bin/clang++"},
            {"RUSTFLAGS": "-C target-cpu=native"},
            {"RUSTC_WRAPPER": "/tmp/wrapper"},
        )
        for bad in bad_values:
            with self.subTest(bad=bad), patch.dict(builder.os.environ, bad, clear=True), patch.object(
                builder, "verify_candidate"
            ), patch.object(builder, "validate_toolchain"):
                with self.assertRaises(builder.G1BuilderError):
                    builder.build_runtime_artifact(
                        repo=ROOT, row_id="g1-gfx1030", reviewed_sha=SHA,
                        tested_sha=SHA, workflow_sha=SHA, tree_oid=TREE,
                        output_dir=self.private_root() / "g1-gfx1030", runner=FakeRunner(),
                    )

    def test_host_stub_is_not_silently_staged(self) -> None:
        fake = FakeRunner(
            binary=b"\x7fELFhost-stub\0sllm_hip_evidence_submit\0",
            host_output=host_readobj_fixture().replace(".hip_fatbin", ".not_fatbin"),
        )
        root = self.private_root()
        with patch.object(builder, "verify_candidate"), patch.object(builder, "validate_toolchain"):
            with self.assertRaises(builder.G1BuilderError):
                self.build(fake=fake, output_root=root)
        self.assertFalse((root / "g1-gfx1030" / builder.BINARY_NAME).exists())

    def test_parser_and_inspector_reject_wrong_target_missing_bundle_and_wrong_features(self) -> None:
        with self.assertRaises(builder.ContractError):
            parse_g1_device_readobj(device_readobj_fixture("gfx1201"), "gfx1030")
        with self.assertRaises(builder.ContractError):
            parse_g1_device_readobj(device_readobj_fixture("gfx1030", wavefront_size=64), "gfx1030")
        with self.assertRaises(builder.ContractError):
            parse_g1_device_readobj(device_readobj_fixture("gfx1030", flags="4e"), "gfx1030")
        root = self.private_root()
        artifact = root / builder.BINARY_NAME
        artifact.write_bytes(b"fixture")
        artifact.chmod(0o700)
        with self.assertRaises(builder.ContractError):
            inspect_g1_runtime_artifact(
                artifact,
                "gfx1030",
                tool_runner=FakeRunner(bundles=["host-x86_64-unknown-linux-gnu-"]),
            )
        with self.assertRaises(builder.ContractError):
            inspect_g1_runtime_artifact(
                artifact,
                "gfx1030",
                tool_runner=FakeRunner(device_output=device_readobj_fixture("gfx1201")),
            )

    def test_cargo_failure_does_not_leave_partial_row_artifact(self) -> None:
        class FailingRunner(FakeRunner):
            def __call__(self, argv, **kwargs):
                command = tuple(str(item) for item in argv)
                if command[:3] == ("cargo", "+1.97.1", "build"):
                    raise builder.G1BuilderError("compiler failed")
                return super().__call__(argv, **kwargs)

        root = self.private_root()
        with patch.object(builder, "verify_candidate"), patch.object(builder, "validate_toolchain"):
            with self.assertRaises(builder.G1BuilderError):
                builder.build_runtime_artifact(
                    repo=ROOT, row_id="g1-gfx1030", reviewed_sha=SHA,
                    tested_sha=SHA, workflow_sha=SHA, tree_oid=TREE,
                    output_dir=root / "g1-gfx1030", runner=FailingRunner(),
                )
        self.assertFalse((root / "g1-gfx1030").exists())
        self.assertFalse((root / "target").exists())

    def test_run_argv_has_no_shell_and_rejects_unbounded_timeout(self) -> None:
        with self.assertRaises(builder.G1BuilderError):
            builder.run_argv(["true"], timeout=builder.MAX_BUILD_TIMEOUT_SECONDS + 1)
        source = Path(builder.__file__).read_text(encoding="utf-8")
        self.assertIn("shell=False", source)
        self.assertNotIn("communicate(", source)

    def test_run_argv_fails_closed_on_bounded_output_overflow(self) -> None:
        with self.assertRaises(builder.G1BuilderError):
            builder.run_argv(
                [sys.executable, "-c", "import sys; sys.stderr.write('x' * (1024 * 1024 + 1))"],
                timeout=5.0,
            )

    def test_run_argv_terminates_and_reaps_process_group_on_timeout(self) -> None:
        command = [
            sys.executable,
            "-c",
            "import subprocess,sys,time; subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); time.sleep(30)",
        ]
        started = time.monotonic()
        with self.assertRaises(builder.G1BuilderError):
            builder.run_argv(command, timeout=0.1)
        self.assertLess(time.monotonic() - started, 5.0)

    def test_toolchain_environment_rejects_wrong_values_and_accepts_exact_tuple(self) -> None:
        with patch.dict(builder.os.environ, {"CMAKE_HIP_ARCHITECTURES": "gfx1030"}, clear=True):
            builder._validate_toolchain_env("gfx1030", Path("/opt/rocm"))
        with patch.dict(builder.os.environ, {"CMAKE_HIP_ARCHITECTURES": "gfx12-generic"}, clear=True):
            with self.assertRaises(builder.G1BuilderError):
                builder._validate_toolchain_env("gfx1030", Path("/opt/rocm"))


if __name__ == "__main__":
    unittest.main()
