#!/usr/bin/env python3
"""Production controller for canonical semantic RMSNorm G1 evidence.

Only this controller can turn fixed-worker raw frames and controller-held
sealed descriptors into an aggregate.  It deliberately has no runner-script,
interpreter, process-factory, or report-import injection surface.
"""

from __future__ import annotations

import sys


_FRESH_PYTHON_BASELINE = {
    "__future__", "__main__", "_abc", "_codecs", "_frozen_importlib", "_frozen_importlib_external",
    "_imp", "_io", "_signal", "_thread", "_warnings", "_weakref", "abc", "builtins", "codecs",
    "encodings", "encodings.aliases", "encodings.utf_8", "io", "marshal", "posix", "sys", "time",
    "zipimport",
}
_BOOTSTRAP_SOURCES = (
    "Cargo.toml",
    "Cargo.lock",
    "crates/sllm-core/Cargo.toml",
    "crates/sllm-core/src/backend.rs",
    "crates/sllm-core/src/dtype.rs",
    "crates/sllm-core/src/execution.rs",
    "crates/sllm-core/src/fake.rs",
    "crates/sllm-core/src/handles.rs",
    "crates/sllm-core/src/lib.rs",
    "crates/sllm-core/src/model.rs",
    "crates/sllm-core/src/op.rs",
    "crates/sllm-core/src/registry.rs",
    "crates/sllm-core/src/tensor.rs",
    "crates/sllm-hip-sys/Cargo.toml",
    "crates/sllm-hip-sys/build.rs",
    "crates/sllm-hip-sys/src/bindings.rs",
    "crates/sllm-hip-sys/src/evidence_bindings.rs",
    "crates/sllm-hip-sys/src/lib.rs",
    "crates/sllm-hip/Cargo.toml",
    "crates/sllm-hip/src/bin/sllm-hip-evidence.rs",
    "crates/sllm-hip/src/lib.rs",
    "crates/sllm-hip/src/bridge.rs",
    "crates/sllm-hip/src/rmsnorm.rs",
    "crates/sllm-hip/src/runtime.rs",
    "crates/sllm-hip/src/bin/sllm-rmsnorm-g1-evidence.rs",
    "include/sllm/hip.h",
    "include/sllm/sllm.h",
    "native/hip/CMakeLists.txt",
    "native/hip/src/abi_layout_probe.cpp",
    "native/hip/src/evidence_abi.h",
    "native/hip/src/header_c_compile.c",
    "native/hip/src/header_cpp_compile.cpp",
    "native/hip/src/hip_compile_probe.hip.cpp",
    "native/hip/src/hip_evidence_runtime.hip.cpp",
    "native/hip/src/hip_evidence_stub.cpp",
    "native/hip/src/hip_stub.cpp",
    "native/hip/src/public_runtime.hip.cpp",
    "native/hip/src/public_runtime_internal.hpp",
    "native/hip/src/public_runtime_stub.cpp",
    "native/hip/src/rmsnorm_api.cpp",
    "native/hip/src/rmsnorm_api.hpp",
    "native/hip/src/rmsnorm_kernel.hip.cpp",
    "native/hip/src/rmsnorm_kernel_internal.hpp",
    "docs/models/locks/qwen3.5-4b-bf16.json",
    "ci/tools/orchestrate_rmsnorm_g1_evidence.py",
    "ci/tools/common.py",
    "ci/tools/exact_actions.py",
    "ci/tools/validate_rmsnorm_g1_contracts.py",
    "ci/tools/build_rmsnorm_g1_runtime.py",
    "ci/tools/run_rmsnorm_g1_runtime.py",
    "ci/tools/validate_g0_contracts.py",
    "ci/tools/run_g0_preflight.py",
    "ci/tools/validate_h3_contracts.py",
    "ci/matrix/rmsnorm-semantic-g1-v1.json",
    "ci/schema/rmsnorm-semantic-g1-matrix-v1.schema.json",
    "ci/schema/rmsnorm-semantic-g1-artifact-v1.schema.json",
    "ci/schema/rmsnorm-semantic-g1-report-v1.schema.json",
    "ci/schema/rmsnorm-semantic-g1-aggregate-v1.schema.json",
    ".github/workflows/semantic-rmsnorm-g1.yml",
)
_CONTROLLER_PYTHON_SHA256 = "1643dacd9feaedc58f3cc581e4d22577dfe25c09b10282936186ccf0f2e61118"
_CONTROLLER_PYTHON_SIZE = 8_020_928


def _fresh_controller_gate() -> tuple[str, dict[str, bytes]]:
    """Authenticate a genuine, isolated, direct controller process first.

    This function intentionally uses only the interpreter's preloaded stdlib
    and fixed absolute OS paths.  It runs before a project module can be
    imported, so ``runpy``, same-process preloading, a caller ``PYTHONPATH``,
    and a mutable copied checkout cannot become controller authority.
    """

    if not sys.flags.isolated or not sys.flags.no_site or sys.executable != "/usr/bin/python3":
        raise RuntimeError("controller requires fixed /usr/bin/python3 -I -S")
    unexpected = set(sys.modules) - _FRESH_PYTHON_BASELINE
    if unexpected:
        raise RuntimeError("fresh controller rejects preloaded modules: " + ", ".join(sorted(unexpected)))
    import fcntl
    import hashlib
    import os
    import stat as _stat
    import subprocess as _subprocess
    from pathlib import Path as _Path

    workspace_text = os.environ.get("GITHUB_WORKSPACE", "")
    if not workspace_text:
        raise RuntimeError("controller requires GITHUB_WORKSPACE")
    workspace = _Path(workspace_text)
    if not workspace.is_absolute() or "\x00" in workspace_text:
        raise RuntimeError("controller workspace path is not closed and absolute")
    resolved_workspace = workspace.resolve(strict=True)
    if str(workspace) != str(resolved_workspace) or workspace.is_symlink() or not resolved_workspace.is_dir():
        raise RuntimeError("controller workspace is not an exact non-symlink path")
    controller_fd_text = os.environ.get("SLLM_G1_CONTROLLER_FD", "")
    try:
        controller_fd = int(controller_fd_text)
        controller_details = os.fstat(controller_fd)
        controller_seals = fcntl.fcntl(controller_fd, fcntl.F_GET_SEALS)
    except (OSError, ValueError) as exc:
        raise RuntimeError("controller requires an inherited sealed controller-source descriptor") from exc
    required_seals = (
        fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_GROW | fcntl.F_SEAL_WRITE | fcntl.F_SEAL_SEAL
    )
    if (
        controller_fd < 3
        or not _stat.S_ISREG(controller_details.st_mode)
        or controller_details.st_size < 1
        or controller_details.st_size > 4 * 1024 * 1024
        or controller_seals & required_seals != required_seals
    ):
        raise RuntimeError("controller source descriptor is not a bounded fully sealed immutable file")
    expected_script = f"/proc/self/fd/{controller_fd}"
    try:
        cmdline = _Path("/proc/self/cmdline").read_bytes().split(b"\0")[:-1]
        live_executable = os.stat("/proc/self/exe")
        fixed_executable = os.stat("/usr/bin/python3")
    except OSError as exc:
        raise RuntimeError("controller cannot inspect its direct Linux process identity") from exc
    if cmdline[:4] != [b"/usr/bin/python3", b"-I", b"-S", expected_script.encode("ascii")]:
        raise RuntimeError("controller rejects non-direct, runpy, wrong-executable, or non-isolated argv")
    if (live_executable.st_dev, live_executable.st_ino) != (fixed_executable.st_dev, fixed_executable.st_ino):
        raise RuntimeError("controller /proc executable is not /usr/bin/python3")
    python_bytes = _Path("/usr/bin/python3").read_bytes()
    if len(python_bytes) != _CONTROLLER_PYTHON_SIZE or hashlib.sha256(python_bytes).hexdigest() != _CONTROLLER_PYTHON_SHA256:
        raise RuntimeError("controller fixed Python bytes differ from the reviewed executable pin")
    allowed_environment = {
        "PATH", "LC_CTYPE", "HOME", "CI", "GITHUB_ACTIONS", "GITHUB_SHA", "GITHUB_WORKSPACE", "RUNNER_TEMP", "RUN_ROOT",
        "REVIEWED_SHA", "TESTED_SHA", "WORKFLOW_SHA", "GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT", "GITHUB_WORKFLOW",
        "SLLM_G1_CONTROLLER_FD",
    }
    if set(os.environ) != allowed_environment:
        raise RuntimeError("controller environment is not the exact closed workflow environment")
    if os.environ.get("PATH") != "/usr/bin:/bin" or os.environ.get("LC_CTYPE") != "C.UTF-8" or os.environ.get("CI") != "true" or os.environ.get("GITHUB_ACTIONS") != "true" or os.environ.get("GITHUB_WORKFLOW") != "semantic-rmsnorm-g1":
        raise RuntimeError("controller fixed workflow environment drifted")
    reviewed = os.environ.get("REVIEWED_SHA", "")
    if len(reviewed) not in (40, 64) or any(value not in "0123456789abcdef" for value in reviewed):
        raise RuntimeError("controller reviewed SHA is malformed")
    if any(os.environ.get(name) != reviewed for name in ("GITHUB_SHA", "TESTED_SHA", "WORKFLOW_SHA")):
        raise RuntimeError("controller workflow candidate SHA inputs do not exactly agree")
    if not os.environ.get("HOME", "").startswith("/") or not os.environ.get("RUNNER_TEMP", "").startswith("/") or not os.environ.get("RUN_ROOT", "").startswith("/"):
        raise RuntimeError("controller private workflow paths are not absolute")
    for name in ("GITHUB_RUN_ID", "GITHUB_RUN_ATTEMPT"):
        if not os.environ.get(name, "").isdigit() or int(os.environ[name]) < 1:
            raise RuntimeError("controller workflow run identity is malformed")
    expected_argv = [
        b"/usr/bin/python3",
        b"-I",
        b"-S",
        expected_script.encode("ascii"),
        b"--artifact-root",
        (os.environ["RUN_ROOT"] + "/artifacts").encode("utf-8"),
        b"--output-dir",
        (os.environ["RUN_ROOT"] + f"/rmsnorm-semantic-g1-aggregate-{os.environ['GITHUB_RUN_ID']}-{os.environ['GITHUB_RUN_ATTEMPT']}").encode("utf-8"),
        b"--run-id",
        os.environ["GITHUB_RUN_ID"].encode("ascii"),
        b"--run-attempt",
        os.environ["GITHUB_RUN_ATTEMPT"].encode("ascii"),
        b"--reviewed-sha",
        reviewed.encode("ascii"),
        b"--tested-sha",
        reviewed.encode("ascii"),
        b"--workflow-sha",
        reviewed.encode("ascii"),
    ]
    if cmdline != expected_argv:
        raise RuntimeError("controller argv is not the exact reviewed workflow invocation")
    git_environment = {
        "PATH": "/usr/bin:/bin", "LC_ALL": "C", "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_SYSTEM": "/dev/null", "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_COUNT": "0",
        "GIT_NO_REPLACE_OBJECTS": "1",
    }
    def git_bytes(*arguments: str) -> bytes:
        completed = _subprocess.run(
            ["/usr/bin/git", "--no-replace-objects", "-C", str(resolved_workspace), *arguments],
            stdin=_subprocess.DEVNULL, stdout=_subprocess.PIPE, stderr=_subprocess.PIPE,
            env=git_environment, check=False, timeout=30.0,
        )
        if completed.returncode != 0:
            raise RuntimeError("controller cannot read the reviewed immutable Git object")
        return completed.stdout
    object_format = git_bytes("rev-parse", "--show-object-format=storage").decode("ascii").strip()
    oid_width = {"sha1": 40, "sha256": 64}.get(object_format)
    if oid_width is None or len(reviewed) != oid_width:
        raise RuntimeError("controller Git object format or candidate OID width is unsupported")
    if git_bytes("for-each-ref", "--format=%(refname)", "refs/replace"):
        raise RuntimeError("controller rejects Git replacement refs")
    config_bytes = git_bytes("config", "--local", "--null", "--list")
    safe_config = {"core.repositoryformatversion", "core.filemode", "core.bare", "core.logallrefupdates", "core.symlinks", "core.ignorecase", "core.precomposeunicode", "extensions.objectformat"}
    config_keys = [part.split(b"\n", 1)[0].decode("ascii") for part in config_bytes.split(b"\0") if part]
    if any(key not in safe_config for key in config_keys):
        raise RuntimeError("controller rejects dangerous Git local configuration")
    if git_bytes("rev-parse", "--verify", "HEAD^{commit}").decode("ascii").strip() != reviewed:
        raise RuntimeError("controller checkout HEAD does not equal the reviewed immutable SHA")
    recomputed_tree = git_bytes("rev-parse", "--verify", "HEAD^{tree}").decode("ascii").strip()
    if len(recomputed_tree) != oid_width:
        raise RuntimeError("controller recomputed tree OID width is invalid")
    if git_bytes("status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("controller refuses a dirty or copied mutable checkout")
    reviewed_sources: dict[str, bytes] = {}
    for relative in _BOOTSTRAP_SOURCES:
        reviewed_bytes = git_bytes("show", f"{reviewed}:{relative}")
        reviewed_sources[relative] = reviewed_bytes
        candidate = resolved_workspace / relative
        try:
            details = candidate.stat()
            current_bytes = candidate.read_bytes()
        except OSError as exc:
            raise RuntimeError("controller reviewed dependency is unavailable") from exc
        if candidate.is_symlink() or not _stat.S_ISREG(details.st_mode) or current_bytes != reviewed_bytes:
            raise RuntimeError("controller/dependency/workflow/matrix/schema bytes differ from the reviewed immutable Git object")
    controller_bytes = os.pread(controller_fd, controller_details.st_size, 0)
    if controller_bytes != reviewed_sources["ci/tools/orchestrate_rmsnorm_g1_evidence.py"]:
        raise RuntimeError("executed controller source descriptor does not equal the reviewed immutable Git object")
    return str(resolved_workspace), reviewed_sources


if __name__ == "__main__":
    try:
        _CONTROLLER_WORKSPACE, _CONTROLLER_REVIEWED_SOURCES = _fresh_controller_gate()
    except BaseException as _bootstrap_error:
        print(f"semantic RMSNorm G1 controller: FAIL-CLOSED: {_bootstrap_error}", file=sys.stderr)
        raise SystemExit(2)
    import argparse
    import base64
    import json
    import math
    import os
    import secrets
    import selectors
    import stat
    import struct
    import subprocess
    import time
    import types
    from datetime import datetime, timezone
    from pathlib import Path
    from typing import Any, Mapping

    def _load_reviewed_module(module_name: str, relative: str) -> Any:
        """Execute only reviewed Git-object bytes, never a live checkout path."""

        source = _CONTROLLER_REVIEWED_SOURCES.get(relative)
        if source is None:
            raise RuntimeError(f"controller reviewed module source is missing: {relative}")
        module = types.ModuleType(module_name)
        module.__file__ = f"/controller-sealed/{relative}"
        # Several shared helpers conditionally add their source directory to
        # sys.path only for standalone invocation.  A nonempty synthetic
        # package prevents that mutable-checkout import route here.
        module.__package__ = "sllm_semantic_g1_sealed"
        module.__cached__ = None
        sys.modules[module_name] = module
        exec(compile(source, module.__file__, "exec"), module.__dict__)
        return module

    _sealed_common = _load_reviewed_module("common", "ci/tools/common.py")
    # ``common`` is shared with standalone tools and derives default paths
    # from __file__.  Those defaults are never authority here; bind them to
    # the already-verified workflow workspace before dependent modules load.
    _sealed_common.ROOT = Path(_CONTROLLER_WORKSPACE)
    _sealed_common.SCHEMA_DIR = _sealed_common.ROOT / "ci" / "schema"
    _sealed_common.MATRIX_DIR = _sealed_common.ROOT / "ci" / "matrix"
    _load_reviewed_module("exact_actions", "ci/tools/exact_actions.py")
    _load_reviewed_module("validate_h3_contracts", "ci/tools/validate_h3_contracts.py")
    _load_reviewed_module("validate_g0_contracts", "ci/tools/validate_g0_contracts.py")
    _load_reviewed_module("run_g0_preflight", "ci/tools/run_g0_preflight.py")
    runner = _load_reviewed_module("run_rmsnorm_g1_runtime", "ci/tools/run_rmsnorm_g1_runtime.py")
    contracts = _load_reviewed_module("validate_rmsnorm_g1_contracts", "ci/tools/validate_rmsnorm_g1_contracts.py")
    # Matrix, schemas, and the workflow are evidence authority too.  Keep
    # their reviewed Git bytes in the controller module, rather than allowing
    # any later validator call to reopen a mutable workspace path.
    contracts.bind_controller_reviewed_sources(Path(_CONTROLLER_WORKSPACE), _CONTROLLER_REVIEWED_SOURCES)
    builder = _load_reviewed_module("build_rmsnorm_g1_runtime", "ci/tools/build_rmsnorm_g1_runtime.py")
    ContractError = _sealed_common.ContractError
    ROOT = _sealed_common.ROOT
else:
    # Imported code has no project dependencies and no execution endpoint.
    ContractError = ValueError


class ControllerError(ContractError):
    """A fail-closed controller lifecycle violation."""


_SEALED_WORKER_BOOTSTRAP = """\
import os
import sys
import types

def _read_sealed(fd, label):
    size = os.fstat(fd).st_size
    if size < 1 or size > 4 * 1024 * 1024:
        raise RuntimeError(label + " source descriptor is outside the fixed bound")
    data = os.pread(fd, size, 0)
    if len(data) != size:
        raise RuntimeError(label + " source descriptor changed during read")
    return data

def _load_module(name, fd, filename):
    module = types.ModuleType(name)
    module.__file__ = filename
    module.__package__ = "sealed"
    module.__cached__ = None
    sys.modules[name] = module
    exec(compile(_read_sealed(fd, name), filename, "exec"), module.__dict__)
    return module

if not sys.flags.isolated or len(sys.argv) < 4:
    raise RuntimeError("sealed worker bootstrap arguments are incomplete")
runner_fd = int(sys.argv[1])
runner_path = sys.argv[2]
worker_args = sys.argv[3:]
runner = types.ModuleType("__main__")
runner.__file__ = runner_path
runner.__package__ = "sealed"
runner.__cached__ = None
sys.modules["__main__"] = runner
sys.argv = [runner_path, *worker_args]
exec(compile(_read_sealed(runner_fd, "runner"), runner_path, "exec"), runner.__dict__)
"""


def _iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _private_directory(path: Path, label: str) -> None:
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise ControllerError(f"{label} is not an absolute non-symlink directory")
    details = path.stat()
    if details.st_uid != os.getuid() or stat.S_IMODE(details.st_mode) & 0o077:
        raise ControllerError(f"{label} is not private to the controller user")


def _new_private_directory(path: Path, label: str) -> None:
    if path.exists() or path.is_symlink():
        raise ControllerError(f"{label} already exists")
    try:
        path.mkdir(mode=0o700)
    except OSError as exc:
        raise ControllerError(f"cannot create {label}") from exc
    _private_directory(path, label)


def _independent_facts(row: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read pre/post GPU health and process facts outside the worker process."""

    from run_g0_preflight import (  # noqa: PLC0415
        AMD_SMI_EXECUTABLE,
        amd_smi_list_json,
        nonblocking_host_lock,
        observe_health,
        observe_processes,
    )

    device = {
        "bdf": row["bdf"],
        "uuid": row["uuid"],
        "target": row["target"],
        "physical_hip_index": row["physical_hip_index"],
        "logical_device_index": row["logical_device_index"],
    }
    with nonblocking_host_lock(Path("/tmp/sllm-g0.lock")):
        binding = amd_smi_list_json(dict(row), executable=AMD_SMI_EXECUTABLE)
        if binding.get("hip_id") != row["physical_hip_index"] or binding.get("bdf") != row["bdf"] or binding.get("uuid") != row["uuid"]:
            raise ControllerError("controller GPU binding differs from the canonical row")
        health_observation = observe_health(dict(row), binding, amd_smi=AMD_SMI_EXECUTABLE, sysfs_root=Path("/sys/bus/pci/devices"))
        process_observation = observe_processes(dict(row), binding, amd_smi=AMD_SMI_EXECUTABLE)
    facts = health_observation.get("facts")
    if health_observation.get("available") is not True or health_observation.get("reliable") is not True or not isinstance(facts, Mapping):
        raise ControllerError("controller health facts are unavailable")
    temperature, ras = facts.get("temperature_c"), facts.get("ras_uncorrectable_count")
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or isinstance(ras, bool) or not isinstance(ras, int) or ras < 0:
        raise ControllerError("controller health facts are malformed")
    if process_observation.get("available") is not True or process_observation.get("reliable") is not True or process_observation.get("gpu_processes") or process_observation.get("residual_runner_children"):
        raise ControllerError("controller process facts are not clean")
    return (
        {"available": True, "reliable": True, "state": "OK", "device": device, "temperature_c": float(temperature), "ras_uncorrectable_count": ras},
        {"available": True, "reliable": True, "state": "CLEAN", "device": device, "gpu_processes": [], "residual_runner_children": []},
    )


def _case_request(row: Mapping[str, Any], case: Mapping[str, Any]) -> tuple[bytes, bytes, bytes, float]:
    rows, width = int(case["rows"]), int(case["n"])
    activation, raw_scale = bytearray(), bytearray()
    classification = str(case.get("classification", ""))
    nonfinite_input = str(case.get("nonfinite_input", ""))
    special = {"nan": math.nan, "posinf": math.inf, "neginf": -math.inf}.get(classification)
    if special is not None and nonfinite_input not in {"activation", "raw_scale"}:
        raise ControllerError("nonfinite case does not name an explicit input source")
    if special is None and nonfinite_input != "none":
        raise ControllerError("finite case has a nonfinite input source")
    for index in range(rows * width):
        value = special if index == 0 and special is not None and nonfinite_input == "activation" else ((index * 37 + int(row["seed"])) % 257 - 128) / 32.0
        activation.extend(struct.pack("<H", runner._f32_to_bf16(value)))
    for index in range(width):
        value = special if index == 0 and special is not None and nonfinite_input == "raw_scale" else ((index * 19 + int(row["seed"])) % 65 - 32) / 128.0
        raw_scale.extend(struct.pack("<H", runner._f32_to_bf16(value)))
    epsilon = contracts.MODEL_EPSILON
    return runner.encode_request((rows, width), epsilon, bytes(activation), bytes(raw_scale)), bytes(activation), bytes(raw_scale), epsilon


def _numerics(actual: bytes, expected: bytes, *, atol: float, rtol: float) -> dict[str, Any]:
    if len(actual) != len(expected) or len(actual) % 2:
        raise ControllerError("raw runtime output length differs from the controller oracle")
    max_abs, max_rel, nan_count, inf_count = 0.0, 0.0, 0, 0
    for actual_bits, expected_bits in zip(struct.iter_unpack("<H", actual), struct.iter_unpack("<H", expected), strict=True):
        observed = runner._bf16_to_f32(actual_bits[0])
        reference = runner._bf16_to_f32(expected_bits[0])
        if math.isnan(reference) or math.isnan(observed):
            if not (math.isnan(reference) and math.isnan(observed)):
                raise ControllerError("raw runtime NaN classification differs from the controller oracle")
            nan_count += 1
            continue
        if math.isinf(reference) or math.isinf(observed):
            if not (math.isinf(reference) and math.isinf(observed) and math.copysign(1.0, reference) == math.copysign(1.0, observed)):
                raise ControllerError("raw runtime Inf classification differs from the controller oracle")
            inf_count += 1
            continue
        absolute = abs(observed - reference)
        relative = absolute / abs(reference) if reference else absolute
        max_abs, max_rel = max(max_abs, absolute), max(max_rel, relative)
        if absolute > atol + rtol * abs(reference):
            raise ControllerError("raw runtime output exceeds the pre-registered RMSNorm tolerance")
    return {"tolerance_id": "rmsnorm-bf16-f32-output-v1", "atol": atol, "rtol": rtol, "max_abs_error": max_abs, "max_rel_error": max_rel, "nan_count": nan_count, "inf_count": inf_count}


def _verify_worker_identity(
    pid: int,
    command: list[str],
    interpreter_fd: int,
    runner_fd: int,
    script_witness_fd: int,
) -> dict[str, int]:
    binding = contracts.process_binding(pid)
    try:
        live_exe = os.stat(f"/proc/{pid}/exe")
        retained_exe = os.fstat(interpreter_fd)
        argv = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")[:-1]
    except OSError as exc:
        raise ControllerError("cannot inspect fixed worker identity") from exc
    if (live_exe.st_dev, live_exe.st_ino) != (retained_exe.st_dev, retained_exe.st_ino):
        raise ControllerError("worker executable does not match the controller-pinned interpreter bytes")
    for descriptor, label in (
        (runner_fd, "runner source"),
        (script_witness_fd, "runner source witness"),
    ):
        try:
            live_source = os.stat(f"/proc/{pid}/fd/{descriptor}")
            retained_source = os.fstat(descriptor)
        except OSError as exc:
            raise ControllerError(f"cannot inspect fixed worker {label} descriptor") from exc
        if (live_source.st_dev, live_source.st_ino) != (retained_source.st_dev, retained_source.st_ino):
            raise ControllerError(f"worker {label} does not match controller-pinned immutable bytes")
    try:
        decoded = [part.decode("utf-8") for part in argv]
    except UnicodeDecodeError as exc:
        raise ControllerError("worker argv is not UTF-8") from exc
    if decoded != command:
        raise ControllerError("worker argv does not equal the fixed controller command")
    return binding


def _drain_streams_nonblocking(process: subprocess.Popen[bytes], timeout_seconds: float) -> tuple[bytes, bytes]:
    if process.stdout is None or process.stderr is None:
        raise ControllerError("worker diagnostic pipes are missing")
    fds = {process.stdout.fileno(): bytearray(), process.stderr.fileno(): bytearray()}
    for descriptor in fds:
        os.set_blocking(descriptor, False)
    selector = selectors.DefaultSelector()
    for descriptor in fds:
        selector.register(descriptor, selectors.EVENT_READ)
    deadline = time.monotonic() + timeout_seconds
    try:
        while selector.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ControllerError("worker diagnostic pipes did not close after group cleanup")
            for key, _mask in selector.select(remaining):
                descriptor = int(key.fileobj)
                try:
                    chunk = os.read(descriptor, 65536)
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(descriptor)
                    continue
                fds[descriptor].extend(chunk)
                if len(fds[descriptor]) > contracts.MAX_OUTPUT:
                    raise ControllerError("worker diagnostic output exceeded the bounded limit")
        return tuple(bytes(fds[descriptor]) for descriptor in fds)  # type: ignore[return-value]
    finally:
        selector.close()


def _receive_worker_frame(
    sock: Any,
    binding: Mapping[str, Any],
    deadline: float,
    *,
    phase: str,
    process: subprocess.Popen[bytes],
    containment: Any,
) -> dict[str, Any]:
    """Receive one frame while policing worker output, RSS, and liveness."""

    if process.stdout is None or process.stderr is None:
        raise ControllerError("worker diagnostic pipes are missing")
    selector = selectors.DefaultSelector()
    streams = (process.stdout.fileno(), process.stderr.fileno())
    try:
        selector.register(sock, selectors.EVENT_READ, "controller")
        for descriptor in streams:
            os.set_blocking(descriptor, False)
            selector.register(descriptor, selectors.EVENT_READ, "diagnostic")
        while True:
            if process.poll() is not None:
                raise ControllerError("worker exited before its authenticated controller frame")
            try:
                containment.assert_rss_within(runner.MAX_RUNTIME_RSS_BYTES)
            except (OSError, ValueError) as exc:
                raise ControllerError("worker Linux RSS/containment observation failed") from exc
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ControllerError("worker controller deadline expired")
            events = selector.select(min(remaining, 0.05))
            for key, _mask in events:
                if key.data == "diagnostic":
                    try:
                        chunk = os.read(int(key.fileobj), 65536)
                    except BlockingIOError:
                        continue
                    if chunk:
                        diagnostic = chunk.decode("utf-8", "replace")[:1024].strip()
                        raise ControllerError(f"fixed worker emitted unexpected diagnostic output: {diagnostic}")
                    selector.unregister(key.fileobj)
            if any(key.data == "controller" for key, _mask in events):
                try:
                    document, credentials = contracts.ipc_recv(sock)
                except (OSError, ContractError) as exc:
                    for _attempt in range(10):
                        worker_status = process.poll()
                        if worker_status is not None:
                            break
                        time.sleep(0.002)
                    raise ControllerError(
                        f"cannot receive authenticated worker {phase} frame: {exc}; worker_status={worker_status}"
                    ) from exc
                if credentials != (binding["pid"], binding["uid"], binding["gid"]):
                    raise ControllerError("worker frame kernel credentials do not match its PID/starttime/UID/GID binding")
                return document
    finally:
        selector.close()


def _run_row(*, repo: Path, artifact_root: Path, row: Mapping[str, Any], run_id: str, run_attempt: int, identity: Mapping[str, Any], authority: Mapping[str, Any]) -> Any:
    result = builder.build_runtime_artifact(repo=repo, row_id=str(row["row_id"]), identity=identity, authority=authority, output_dir=artifact_root / str(row["row_id"]), timeout_seconds=float(row["timeout_seconds"]))
    bundle = contracts.capture_builder_bundle(result, row=row, identity=identity, repo=repo, authority=authority)
    artifact_facts = {
        "metadata_sha256": bundle.metadata.record["sha256"],
        "binary_sha256": bundle.binary.record["sha256"],
        "companion_sha256": bundle.companion.record["sha256"],
        "loader_sha256": bundle.loader.record["sha256"],
        "runtime_library_sha256": contracts.sha256_json([library.record["sha256"] for library in bundle.libraries]),
        "runtime_dependency_closure_sha256": str(bundle.metadata_document["runtime_dependency_closure"]["sha256"]),
        "compiler_execution_sha256": contracts.sha256_json(bundle.metadata_document["compiler_execution"]),
    }
    interpreter = contracts.approved_python_interpreter()
    script = contracts.approved_repository_file(repo, identity, contracts.RUNNER_RELATIVE_PATH, "fixed semantic G1 worker")
    script_witness = os.dup(script.fd)
    parent_socket, child_socket = contracts.controller_socketpair()
    process: subprocess.Popen[bytes] | None = None
    containment: runner.LinuxContainment | None = None
    worker_binding: dict[str, int] | None = None
    health_pre: dict[str, Any] | None = None
    process_pre: dict[str, Any] | None = None
    health_post: dict[str, Any] | None = None
    process_post: dict[str, Any] | None = None
    case_documents: list[dict[str, Any]] = []
    raw_snapshots: list[contracts.SealedDescriptor] = []
    candidate_document = contracts.verify_repository_identity(repo, identity)
    candidate_digest = contracts.sha256_json(candidate_document)
    started_wall = _iso()
    started_monotonic_ns = time.monotonic_ns()
    try:
        health_pre, process_pre = _independent_facts(row)
        child_fd = child_socket.fileno()
        command = [
            str(contracts.fd_path(interpreter.fd)), "-I", "-S", "-c", _SEALED_WORKER_BOOTSTRAP,
            str(script.fd), str(repo / contracts.RUNNER_RELATIVE_PATH),
            "--worker",
            "--controller-fd", str(child_fd), "--controller-pid", str(os.getpid()),
            "--row", str(row["row_id"]), "--target", str(row["target"]),
            "--physical-hip-index", str(row["physical_hip_index"]), "--timeout-seconds", str(row["timeout_seconds"]),
            "--loader-fd", str(bundle.loader.fd), "--executable-fd", str(bundle.binary.fd),
            *sum((["--library-fd", str(library.fd)] for library in bundle.libraries), []),
        ]
        pass_fds = (child_fd, interpreter.fd, script.fd, script_witness, bundle.loader.fd, bundle.binary.fd, *(library.fd for library in bundle.libraries))
        parent_identities = runner._parent_open_identities()
        containment = runner.LinuxContainment.begin()
        process = subprocess.Popen(command, cwd="/", stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True, close_fds=True, pass_fds=pass_fds, env={"PATH": "/usr/bin:/bin"})
        # start_new_session makes the leader PID the immutable process-group
        # identifier even if the direct child exits before cleanup starts.
        containment.bind_root(process.pid, process.pid)
        worker_binding = _verify_worker_identity(process.pid, command, interpreter.fd, script.fd, script_witness)
        runner.audit_child_fd_inheritance(
            process.pid,
            (child_fd, script.fd, script_witness, bundle.loader.fd, bundle.binary.fd, *(library.fd for library in bundle.libraries)),
            expected_binding=worker_binding,
            optional_descriptors=(interpreter.fd,),
            forbidden_descriptors=(parent_socket.fileno(), bundle.metadata.fd, bundle.companion.fd, *(item.fd for item in bundle.sidecars)),
            parent_identities=parent_identities,
        )
        child_socket.close()
        child_socket = None  # type: ignore[assignment]
        row_deadline = time.monotonic() + float(row["timeout_seconds"])
        ready = _receive_worker_frame(parent_socket, worker_binding, row_deadline, phase="ready", process=process, containment=containment)
        if set(ready) != {"kind", "binding"} or ready.get("kind") != "ready" or ready.get("binding") != worker_binding:
            raise ControllerError("worker ready frame does not bind its observed process identity")
        challenge = base64.b64encode(secrets.token_bytes(32)).decode("ascii")
        contracts.ipc_send(parent_socket, {"kind": "start", "row_id": row["row_id"], "target": row["target"], "challenge": challenge})
        for order, case in enumerate(contracts.EXPECTED_CASES):
            request, activation, raw_scale, epsilon = _case_request(row, case)
            case_started_wall, case_started_monotonic_ns = _iso(), time.monotonic_ns()
            contracts.ipc_send(parent_socket, {"kind": "case", "challenge": challenge, "order": order, "request_b64": base64.b64encode(request).decode("ascii")})
            frame = _receive_worker_frame(parent_socket, worker_binding, row_deadline, phase=f"case-{order}", process=process, containment=containment)
            if frame.get("kind") == "failure":
                raise ControllerError(f"fixed worker rejected raw runtime execution: {frame.get('error', 'unknown failure')}")
            if set(frame) != {"kind", "challenge", "order", "response_b64", "stderr_b64"} or frame.get("kind") != "raw-case" or frame.get("challenge") != challenge or frame.get("order") != order:
                raise ControllerError("worker raw-case frame is not the expected one-shot response")
            try:
                response = base64.b64decode(str(frame["response_b64"]).encode("ascii"), validate=True)
                stderr = base64.b64decode(str(frame["stderr_b64"]).encode("ascii"), validate=True)
            except (UnicodeError, ValueError) as exc:
                raise ControllerError("worker raw-case frame has invalid base64") from exc
            if len(response) > contracts.MAX_OUTPUT or stderr:
                raise ControllerError("worker raw runtime output is oversized or has unexpected stderr")
            parsed = runner.parse_response(response, expected_target=str(row["target"]), expected_device_index=int(row["logical_device_index"]), expected_shape=(int(case["rows"]), int(case["n"])), expected_epsilon=epsilon)
            raw_snapshots.append(contracts.snapshot_bytes(
                response,
                logical_path=f"/controller/raw/{row['row_id']}/case-{order}.bin",
                label=f"raw-{row['target']}-{order}",
            ))
            expected_output = runner.independent_rmsnorm_oracle(activation, raw_scale, int(case["rows"]), int(case["n"]), epsilon)
            numeric = _numerics(parsed["output"], expected_output, atol=0.0078125, rtol=0.015625)
            finished_monotonic_ns, finished_wall = time.monotonic_ns(), _iso()
            case_documents.append({
                "order": order,
                "id": case["id"],
                "rows": case["rows"],
                "n": case["n"],
                "classification": case["classification"],
                "nonfinite_input": case["nonfinite_input"],
                "request_sha256": contracts.sha256_bytes(request),
                "response_sha256": contracts.sha256_bytes(response),
                "response_evidence": {
                    "path": f"rows/{row['row_id']}/raw/case-{order}.bin",
                    "sidecar_path": f"rows/{row['row_id']}/raw/case-{order}.bin.sha256",
                    "size_bytes": len(response),
                    "sha256": contracts.sha256_bytes(response),
                    "sidecar_sha256": contracts.sha256_bytes(contracts._sidecar_text(contracts.sha256_bytes(response), f"case-{order}.bin")),
                    "candidate_sha256": candidate_digest,
                    "row_id": row["row_id"],
                    "case_id": case["id"],
                    "order": order,
                },
                "resource_counts": parsed["resource_counts"],
                "dispatch_id": parsed["dispatch_id"],
                "dispatch_count": parsed["dispatch_count"],
                "kernel_symbol": parsed["kernel_symbol"],
                "device_symbol": parsed["device_symbol"],
                "numerics": numeric,
                "controller_started_at": case_started_wall,
                "controller_finished_at": finished_wall,
                "controller_duration_ns": finished_monotonic_ns - case_started_monotonic_ns,
            })
        contracts.ipc_send(parent_socket, {"kind": "finish", "challenge": challenge})
        done = _receive_worker_frame(parent_socket, worker_binding, row_deadline, phase="done", process=process, containment=containment)
        if done != {"kind": "done", "challenge": challenge, "case_count": len(contracts.EXPECTED_CASES)}:
            raise ControllerError("worker completion frame is malformed")
    except BaseException:
        for frame in raw_snapshots:
            frame.close()
        raise
    finally:
        cleanup_error: ControllerError | None = None
        try:
            if process is not None:
                if containment is None or not containment.terminate_and_reap(process):
                    cleanup_error = ControllerError("controller could not prove complete worker Linux containment/reaping")
                try:
                    stdout, stderr = _drain_streams_nonblocking(process, 2.0)
                    if stdout or stderr:
                        cleanup_error = cleanup_error or ControllerError("fixed worker emitted unexpected diagnostic stream output")
                except (ControllerError, OSError, ValueError) as exc:
                    cleanup_error = cleanup_error or ControllerError("worker diagnostic pipes did not close cleanly")
                    if cleanup_error.__cause__ is None:
                        cleanup_error.__cause__ = exc
        finally:
            if process is not None:
                runner.close_process_streams(process)
            try:
                parent_socket.close()
            finally:
                if child_socket is not None:
                    child_socket.close()
                script.close()
                interpreter.close()
                try:
                    os.close(script_witness)
                except OSError:
                    pass
                bundle.close()
        try:
            # These are independently observed even when the worker or raw
            # protocol failed; no worker-owned facts can turn a failed run
            # into a clean postcondition.
            health_post, process_post = _independent_facts(row)
        except (ContractError, OSError, TypeError, ValueError) as exc:
            cleanup_error = cleanup_error or ControllerError("controller could not collect independent post-run facts")
            if cleanup_error.__cause__ is None:
                raise cleanup_error from exc
        if cleanup_error is not None:
            raise cleanup_error
    if health_pre is None or process_pre is None or health_post is None or process_post is None or worker_binding is None:
        raise ControllerError("controller did not collect pre-run facts and worker identity")
    finished_monotonic_ns, finished_wall = time.monotonic_ns(), _iso()
    raw_frames = b"".join(contracts.fd_read_all(frame.fd, max_bytes=contracts.MAX_OUTPUT) for frame in raw_snapshots)
    resource_counts = {name: sum(int(case["resource_counts"][name]) for case in case_documents) for name in ("allocation_count", "copy_count", "dispatch_count", "kernel_count")}
    report = {
        "schema_version": "rmsnorm-semantic-g1-report-v1",
        "report_id": f"rmsnorm-semantic-g1-report-{run_id}-{run_attempt}-{row['target']}",
        "row_id": row["row_id"],
        "target": row["target"],
        "state": "PASS",
        "required": True,
        "run_id": run_id,
        "run_attempt": run_attempt,
        "candidate": candidate_document,
        "contracts": contracts.authority_contract_hashes(authority),
        "authority": dict(authority),
        "artifact_kind": "rmsnorm-semantic-g1-runtime",
        "scope": contracts.EXPECTED_SCOPE,
        "device": {key: row[key] for key in ("bdf", "uuid", "target", "physical_hip_index", "logical_device_index")},
        "artifact": {
            **artifact_facts,
        },
        "compiler_execution": bundle.metadata_document["compiler_execution"],
        "worker": {"pid": worker_binding["pid"], "starttime": worker_binding["starttime"], "uid": worker_binding["uid"], "gid": worker_binding["gid"], "script_sha256": script.record["sha256"], "interpreter_sha256": interpreter.record["sha256"]},
        "raw_frame_sha256": contracts.sha256_bytes(raw_frames),
        "resource_counts": resource_counts,
        "cases": case_documents,
        "health_pre": health_pre,
        "health_post": health_post,
        "process_pre": process_pre,
        "process_post": process_post,
        "controller_started_at": started_wall,
        "controller_finished_at": finished_wall,
        "controller_duration_ns": finished_monotonic_ns - started_monotonic_ns,
    }
    try:
        contracts.validate_report_document(
            report,
            row=row,
            identity=identity,
            repo=repo,
            authority=authority,
            artifact_facts=artifact_facts,
        )
        return report, tuple(raw_snapshots), dict(artifact_facts)
    except BaseException:
        for frame in raw_snapshots:
            frame.close()
        raise


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--artifact-root", type=Path, required=True)
    result.add_argument("--output-dir", type=Path, required=True)
    result.add_argument("--run-id", required=True)
    result.add_argument("--run-attempt", type=int, required=True)
    result.add_argument("--reviewed-sha", required=True)
    result.add_argument("--tested-sha", required=True)
    result.add_argument("--workflow-sha", required=True)
    return result


def main(argv: list[str] | None = None) -> int:
    # Importing this module is deliberately non-authoritative.  The emitting
    # implementation is defined only under __main__ below, after canonical
    # checkout and workflow identity checks have succeeded.
    del argv
    print("semantic RMSNorm G1 controller: importable execution is disabled", file=sys.stderr)
    return 2


if __name__ == "__main__":
    def _write_new(path: Path, data: bytes, label: str) -> None:
        if path.exists() or path.is_symlink():
            raise ControllerError(f"refusing to overwrite {label}")
        descriptor = -1
        try:
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, 0o600)
            offset = 0
            while offset < len(data):
                offset += os.write(descriptor, data[offset:])
        except OSError as exc:
            raise ControllerError(f"cannot write {label}") from exc
        finally:
            if descriptor >= 0:
                os.close(descriptor)


    def _write_sidecar(path: Path, label: str) -> None:
        _write_new(path.with_name(path.name + contracts.SIDECAR_SUFFIX), f"{contracts.sha256_file(path)}  {path.name}\n".encode("ascii"), f"{label} sidecar")


    def _recompute_local_row(report: Mapping[str, Any], frames: tuple[Any, ...], artifact_facts: Mapping[str, Any], row: Mapping[str, Any], identity: Mapping[str, Any], repo: Path, authority: Mapping[str, Any]) -> tuple[str, dict[str, int]]:
        """Consume only frames just retained by this process, never imports."""

        contracts.validate_report_document(
            report,
            row=row,
            identity=identity,
            repo=repo,
            authority=authority,
            artifact_facts=artifact_facts,
        )
        cases = report.get("cases")
        if not isinstance(cases, list) or len(frames) != len(cases):
            raise ControllerError("controller retained raw-frame count is incomplete")
        raw_parts: list[bytes] = []
        totals = {name: 0 for name in contracts.EXPECTED_CASE_RESOURCE_COUNTS}
        for order, (expected, case, frame) in enumerate(zip(contracts.EXPECTED_CASES, cases, frames, strict=True)):
            if not isinstance(case, Mapping) or not isinstance(frame, contracts.SealedDescriptor) or frame.fd < 0 or not contracts.descriptor_is_sealed(frame.fd):
                raise ControllerError("controller aggregate received an invalid retained raw frame")
            request, activation, scale, epsilon = _case_request(row, expected)
            raw = contracts.fd_read_all(frame.fd, max_bytes=contracts.MAX_OUTPUT)
            expected_evidence = {
                "path": f"rows/{row['row_id']}/raw/case-{order}.bin",
                "sidecar_path": f"rows/{row['row_id']}/raw/case-{order}.bin.sha256",
                "size_bytes": len(raw),
                "sha256": contracts.sha256_bytes(raw),
                "sidecar_sha256": contracts.sha256_bytes(contracts._sidecar_text(contracts.sha256_bytes(raw), f"case-{order}.bin")),
                "candidate_sha256": contracts.sha256_json(report["candidate"]),
                "row_id": row["row_id"],
                "case_id": expected["id"],
                "order": order,
            }
            if (
                not raw
                or {key: case.get(key) for key in ("order", "id", "rows", "n", "classification", "nonfinite_input")} != {"order": order, **expected}
                or case.get("request_sha256") != contracts.sha256_bytes(request)
                or case.get("response_sha256") != contracts.sha256_bytes(raw)
                or case.get("response_evidence") != expected_evidence
            ):
                raise ControllerError("controller report digest does not recompute from canonical request/raw bytes")
            parsed = runner.parse_response(raw, expected_target=str(row["target"]), expected_device_index=int(row["logical_device_index"]), expected_shape=(int(expected["rows"]), int(expected["n"])), expected_epsilon=epsilon)
            oracle = runner.independent_rmsnorm_oracle(activation, scale, int(expected["rows"]), int(expected["n"]), epsilon)
            numerical = _numerics(parsed["output"], oracle, atol=0.0078125, rtol=0.015625)
            if (
                parsed["resource_counts"] != contracts.EXPECTED_CASE_RESOURCE_COUNTS
                or case.get("resource_counts") != parsed["resource_counts"]
                or any(case.get(key) != parsed[key] for key in ("dispatch_id", "dispatch_count", "kernel_symbol", "device_symbol"))
                or case.get("numerics") != numerical
            ):
                raise ControllerError("controller report response/numerics/dispatch facts are not independently recomputed")
            for name, value in parsed["resource_counts"].items():
                totals[name] += int(value)
            raw_parts.append(raw)
        if totals != contracts.EXPECTED_ROW_RESOURCE_COUNTS:
            raise ControllerError("controller row total resource counts drifted")
        raw_digest = contracts.sha256_bytes(b"".join(raw_parts))
        if report.get("raw_frame_sha256") != raw_digest or report.get("resource_counts") != totals:
            raise ControllerError("controller row aggregate digest/counts are not retained-frame derived")
        artifact = report.get("artifact")
        compiler = report.get("compiler_execution")
        if not isinstance(artifact, Mapping) or not isinstance(compiler, Mapping):
            raise ControllerError("controller report lacks bound artifact/compiler transcript")
        if dict(artifact) != dict(artifact_facts):
            raise ControllerError("controller report artifact bytes do not equal the retained sealed builder bundle")
        if artifact.get("compiler_execution_sha256") != contracts.sha256_json(compiler):
            raise ControllerError("controller report compiler transcript digest does not recompute")
        return raw_digest, totals


    def _emit_only_from_local_controller_rows(rows: list[tuple[dict[str, Any], tuple[Any, ...], dict[str, Any]]], *, repo: Path, output_dir: Path, run_id: str, run_attempt: int, identity: Mapping[str, Any], authority: Mapping[str, Any]) -> None:
        """The sole G1 emission point; not defined when this module is imported."""

        try:
            if len(rows) != len(contracts.ROWS):
                raise ControllerError("controller did not produce both canonical serial rows")
            reports = [item[0] for item in rows]
            if [report.get("row_id") for report in reports] != list(contracts.ROWS):
                raise ControllerError("controller rows are missing, duplicate, or reordered")
            records: list[dict[str, Any]] = []
            matrix = contracts.validate_matrix(repo)
            for (report, frames, artifact_facts), row_id in zip(rows, contracts.ROWS, strict=True):
                row = contracts.row_by_id(matrix, row_id)
                raw_digest, resource_counts = _recompute_local_row(report, frames, artifact_facts, row, identity, repo, authority)
                artifact = report.get("artifact")
                if not isinstance(artifact, Mapping):
                    raise ControllerError("controller row has no artifact binding")
                records.append({
                    "row_id": row_id, "target": row["target"], "state": "PASS",
                    "report_sha256": contracts.sha256_json(report), "binary_sha256": artifact["binary_sha256"],
                    "companion_sha256": artifact["companion_sha256"], "loader_sha256": artifact["loader_sha256"],
                    "runtime_library_sha256": artifact["runtime_library_sha256"],
                    "runtime_dependency_closure_sha256": artifact["runtime_dependency_closure_sha256"],
                    "raw_frame_sha256": raw_digest,
                    "response_evidence": [case["response_evidence"] for case in report["cases"]],
                    "resource_counts": resource_counts,
                    "compiler_execution_sha256": artifact["compiler_execution_sha256"],
                    "compiler_execution": report["compiler_execution"],
                })
            if len({record["binary_sha256"] for record in records}) != 2 or len({record["companion_sha256"] for record in records}) != 2:
                raise ControllerError("canonical target rows did not produce distinct target-qualified artifacts")
            document = {
                "schema_version": "rmsnorm-semantic-g1-aggregate-v1", "aggregate_id": f"rmsnorm-semantic-g1-aggregate-{run_id}-{run_attempt}",
                "suite_id": contracts.MATRIX_SUITE_ID, "tier": "tier_g1", "state": "PASS", "required": True,
                "run_id": run_id, "run_attempt": run_attempt, "candidate": dict(identity), "contracts": contracts.authority_contract_hashes(authority), "authority": dict(authority),
                "artifact_kind": "rmsnorm-semantic-g1-runtime", "expected_rows": list(contracts.ROWS), "rows": records,
                "scope": contracts.EXPECTED_SCOPE, "counts": {"expected_rows": 2, "selected_rows": 2, "collected_rows": 2, "passed_rows": 2, "failed_rows": 0}, "created_at": _iso(),
            }
            contracts.validate_aggregate_document(document, identity=identity, repo=repo, authority=authority)
            if not output_dir.is_absolute() or output_dir.exists() or output_dir.is_symlink() or output_dir.name != f"rmsnorm-semantic-g1-aggregate-{run_id}-{run_attempt}":
                raise ControllerError("aggregate output directory is not a fresh canonical controller path")
            _private_directory(output_dir.parent, "controller aggregate parent")
            output_dir.mkdir(mode=0o700)
            rows_dir = output_dir / "rows"
            rows_dir.mkdir(mode=0o700)
            for report, frames, _artifact_facts in rows:
                row_dir = rows_dir / str(report["row_id"])
                row_dir.mkdir(mode=0o700)
                report_path = row_dir / contracts.REPORT_NAME
                _write_new(report_path, (json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"), "controller row report")
                _write_sidecar(report_path, "controller row report")
                raw_dir = row_dir / "raw"
                raw_dir.mkdir(mode=0o700)
                for order, frame in enumerate(frames):
                    raw_path = raw_dir / f"case-{order}.bin"
                    _write_new(raw_path, contracts.fd_read_all(frame.fd, max_bytes=contracts.MAX_OUTPUT), "controller raw frame")
                    _write_sidecar(raw_path, "controller raw frame")
            aggregate_path = output_dir / "rmsnorm-semantic-g1-aggregate.json"
            _write_new(aggregate_path, (json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8"), "semantic G1 aggregate")
            _write_sidecar(aggregate_path, "semantic G1 aggregate")
        finally:
            for _report, frames, _artifact_facts in rows:
                for frame in frames:
                    if isinstance(frame, contracts.SealedDescriptor):
                        frame.close()


    def _executed_controller_main(argv: list[str] | None = None) -> int:
        args = parser().parse_args(argv)
        try:
            repo = contracts.controller_workspace(Path(os.environ["GITHUB_WORKSPACE"]))
            identity_input = {"reviewed_sha": args.reviewed_sha, "tested_sha": args.tested_sha, "workflow_sha": args.workflow_sha}
            identity = contracts.verify_repository_identity(repo, identity_input)
            authority = contracts.reviewed_authority(repo, identity)
            if (
                contracts.RUN_ID_RE.fullmatch(args.run_id) is None
                or args.run_attempt < 1
                or args.run_id != os.environ["GITHUB_RUN_ID"]
                or str(args.run_attempt) != os.environ["GITHUB_RUN_ATTEMPT"]
            ):
                raise ControllerError("controller run identity is invalid")
            matrix = contracts.validate_matrix(repo)
            rows = [contracts.row_by_id(matrix, row_id) for row_id in contracts.ROWS]
            artifact_root = Path(args.artifact_root)
            run_root = Path(os.environ["RUN_ROOT"])
            if artifact_root != run_root / "artifacts" or Path(args.output_dir) != run_root / f"rmsnorm-semantic-g1-aggregate-{args.run_id}-{args.run_attempt}":
                raise ControllerError("controller artifact/output paths are not the exact closed workflow paths")
            _private_directory(artifact_root.parent, "controller artifact parent")
            _new_private_directory(artifact_root, "controller artifact root")
            local_rows: list[tuple[dict[str, Any], tuple[Any, ...], dict[str, Any]]] = []
            for row in rows:  # canonical order: gfx1030, then gfx1201
                local_rows.append(_run_row(repo=repo, artifact_root=artifact_root, row=row, run_id=args.run_id, run_attempt=args.run_attempt, identity=identity, authority=authority))
            _emit_only_from_local_controller_rows(local_rows, repo=repo, output_dir=Path(args.output_dir), run_id=args.run_id, run_attempt=args.run_attempt, identity=identity, authority=authority)
        except (ControllerError, ContractError, OSError, TypeError, ValueError, subprocess.SubprocessError) as exc:
            print(f"semantic RMSNorm G1 controller: FAIL-CLOSED: {exc}", file=sys.stderr)
            return 2
        print("semantic RMSNorm G1 controller: emitted two controller-owned serial raw-evidence rows")
        return 0


    raise SystemExit(_executed_controller_main())
