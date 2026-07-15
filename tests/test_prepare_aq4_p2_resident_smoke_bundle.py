from __future__ import annotations

import importlib.util
import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator

import pytest


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "benchmarks/results/2026-07-14/qwen35-9b-aq4-production-opt-v0.1/p2/resident-one-case-smoke-prepared-v1"
BINDING = ROOT / "benchmarks/results/2026-07-14/qwen35-9b-aq4-production-opt-v0.1/p2/resident-one-case-smoke-binding-v6"
if (BINDING / "binding-manifest.json").is_file():
    _CHECKED_IN_BINDING_MANIFEST = json.loads(
        (BINDING / "binding-manifest.json").read_text()
    )
    VALIDATOR_COMMIT = _CHECKED_IN_BINDING_MANIFEST["trust_roots"]["validator"]["source_commit"]
    VALIDATOR_SHA = _CHECKED_IN_BINDING_MANIFEST["trust_roots"]["validator"]["sha256"]
else:
    VALIDATOR_COMMIT = ""
    VALIDATOR_SHA = ""
SPEC = importlib.util.spec_from_file_location(
    "aq4_p2_resident_smoke_bundle",
    ROOT / "tools/prepare-aq4-p2-resident-smoke-bundle.py",
)
assert SPEC and SPEC.loader
BUNDLE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUNDLE)


def copy_bundle(tmp_path: Path) -> Path:
    destination = tmp_path / "bundle"
    shutil.copytree(ARTIFACT, destination)
    return destination


def rewrite_json(path: Path, value: dict) -> None:
    path.chmod(0o644)
    path.write_text(json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    path.chmod(0o444)


def fd_identity(descriptor: int) -> dict[str, int]:
    return BUNDLE.named_identity(os.fstat(descriptor))


@contextmanager
def inherited_bundle_map(
    root: Path,
    *,
    mutate: Callable[[dict], None] | None = None,
    sealed: bool = True,
    extra_control: Path | None = None,
    corrupt_hash: bool = False,
) -> Iterator[tuple[int, dict]]:
    descriptors: list[int] = []
    map_descriptor: int | None = None
    try:
        root_descriptor = os.open(
            root,
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0),
        )
        descriptors.append(root_descriptor)
        bindings = [
            {
                "role": "bundle_root",
                "logical_path": str(root),
                "resolved_path": None,
                "descriptor": root_descriptor,
                "kind": "directory",
                "closure": "data_integrity",
                "method": "pre_post_guard",
                "identity": fd_identity(root_descriptor),
                "sha256": None,
            }
        ]
        for name in sorted(set(BUNDLE.REQUIRED_FILES) | {"bundle.json", "SHA256SUMS"}):
            path = root / name
            descriptor = os.open(
                path,
                os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            )
            descriptors.append(descriptor)
            role_name = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
            bindings.append(
                {
                    "role": f"bundle_{role_name}",
                    "logical_path": str(path),
                    "resolved_path": None,
                    "descriptor": descriptor,
                    "kind": "regular_file",
                    "closure": "control_input",
                    "method": "read",
                    "identity": fd_identity(descriptor),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            )
        if extra_control is not None:
            descriptor = os.open(
                extra_control,
                os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
            )
            descriptors.append(descriptor)
            bindings.append(
                {
                    "role": "external_control",
                    "logical_path": str(extra_control),
                    "resolved_path": None,
                    "descriptor": descriptor,
                    "kind": "regular_file",
                    "closure": "control_input",
                    "method": "read",
                    "identity": fd_identity(descriptor),
                    "sha256": hashlib.sha256(extra_control.read_bytes()).hexdigest(),
                }
            )
        value = {
            "schema_version": BUNDLE.FD_MAP_SCHEMA,
            "status": "bound",
            "map_sha256": None,
            "logical_argv_sha256": hashlib.sha256(BUNDLE.canonical(["logical-validator-argv"])).hexdigest(),
            "closure_contract": BUNDLE.FD_CLOSURE_CONTRACT,
            "bindings": bindings,
        }
        if mutate is not None:
            mutate(value)
        if "map_sha256" in value:
            value["map_sha256"] = hashlib.sha256(
                BUNDLE.canonical({**value, "map_sha256": None})
            ).hexdigest()
            if corrupt_hash:
                value["map_sha256"] = "0" * 64
        data = BUNDLE.canonical(value) + b"\n"
        flags = getattr(os, "MFD_CLOEXEC", 0) | getattr(os, "MFD_ALLOW_SEALING", 0)
        map_descriptor = os.memfd_create("validator-test-fd-map", flags)
        os.write(map_descriptor, data)
        if sealed:
            required = (
                fcntl.F_SEAL_SEAL
                | fcntl.F_SEAL_SHRINK
                | fcntl.F_SEAL_GROW
                | fcntl.F_SEAL_WRITE
            )
            fcntl.fcntl(map_descriptor, fcntl.F_ADD_SEALS, required)
        yield map_descriptor, value
    finally:
        if map_descriptor is not None:
            os.close(map_descriptor)
        for descriptor in descriptors:
            os.close(descriptor)


def worker_hardlink_fixture(tmp_path: Path, *, exact_two: bool = True) -> tuple[Path, dict]:
    release = tmp_path / "release"
    deps = release / "deps"
    deps.mkdir(parents=True)
    primary = release / "ullm-aq4-worker"
    alias = deps / "ullm_aq4_worker-03e49ec754c21dc7"
    primary.write_bytes(b"worker-hardlink-fixture")
    primary.chmod(0o755)
    if exact_two:
        os.link(primary, alias)
    metadata = primary.lstat()
    value = {
        "schema_version": "ullm.aq4_p2_resident_worker_link_identity.v2",
        "roots": [str(release), str(deps)],
        "paths": [str(primary), *([str(alias)] if exact_two else [])],
        "primary_path": str(primary),
        "sha256": hashlib.sha256(primary.read_bytes()).hexdigest(),
        "expected": {
            "device": metadata.st_dev,
            "inode": metadata.st_ino,
            "uid": metadata.st_uid,
            "gid": metadata.st_gid,
            "mode": metadata.st_mode,
            "size": metadata.st_size,
            "nlink": metadata.st_nlink,
            "mtime_ns": metadata.st_mtime_ns,
            "ctime_ns": metadata.st_ctime_ns,
        },
    }
    fixture = tmp_path / "worker-hardlinks.json"
    fixture.write_text(json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    fixture.chmod(0o644)
    return fixture, value


def validate_worker_fixture(path: Path, *, hook=None) -> dict:
    return BUNDLE.validate_worker_hardlink_fixture(
        hook=hook,
        fixture_path=path,
        expected_fixture_sha=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def test_active_worker_link_fixture_matches_read_only_production() -> None:
    value = BUNDLE.validate_worker_hardlink_fixture()
    assert value["exact_path_count"] == 1
    assert value["paths"] == [value["primary_path"]]
    assert value["expected"]["nlink"] == 1
    assert value["unknown_hardlinks_possible"] is False
    assert value["sha256"] == BUNDLE.EXPECTED_WORKER_SHA


@pytest.mark.parametrize("exact_two", (False, True))
def test_worker_link_fixture_accepts_declared_exact_topology(tmp_path: Path, exact_two: bool) -> None:
    fixture, value = worker_hardlink_fixture(tmp_path, exact_two=exact_two)
    observed = validate_worker_fixture(fixture)
    assert observed["paths"] == value["paths"]
    assert observed["expected"]["nlink"] == len(value["paths"])


@pytest.mark.parametrize("mutation", ("add", "remove", "different_inode", "content", "mode", "alias_name", "root_escape"))
def test_worker_hardlink_fixture_rejects_initial_mutations(tmp_path: Path, mutation: str) -> None:
    fixture, value = worker_hardlink_fixture(tmp_path)
    primary = Path(value["primary_path"])
    alias = Path(value["paths"][1])
    if mutation == "add":
        os.link(primary, primary.parent / "third-worker-link")
    elif mutation == "remove":
        alias.unlink()
    elif mutation == "different_inode":
        alias.unlink()
        shutil.copy2(primary, alias)
    elif mutation == "content":
        alias.write_bytes(b"changed-worker-content")
    elif mutation == "mode":
        primary.chmod(0o777)
    elif mutation == "alias_name":
        alias.rename(alias.with_name("unexpected-worker-name"))
    else:
        escaped = tmp_path / "escaped-worker"
        alias.rename(escaped)
        value["paths"][1] = str(escaped)
        rewrite_json(fixture, value)
    with pytest.raises(BUNDLE.BundleError, match="worker link"):
        validate_worker_fixture(fixture)


@pytest.mark.parametrize("mutation", ("add", "remove", "content", "primary_swap"))
def test_worker_hardlink_fixture_rejects_late_mutations(tmp_path: Path, mutation: str) -> None:
    fixture, value = worker_hardlink_fixture(tmp_path)
    primary = Path(value["primary_path"])
    alias = Path(value["paths"][1])

    def mutate() -> None:
        if mutation == "add":
            os.link(primary, primary.parent / "late-third-worker-link")
        elif mutation == "remove":
            alias.unlink()
        elif mutation == "content":
            alias.write_bytes(b"late-content-change")
        else:
            moved = primary.with_name("moved-original-worker")
            primary.rename(moved)
            shutil.copy2(moved, primary)

    with pytest.raises((BUNDLE.BundleError, FileNotFoundError), match="worker link|No such file"):
        validate_worker_fixture(fixture, hook=mutate)


def test_worker_hardlink_fixture_rejects_symlinked_deps_root(tmp_path: Path) -> None:
    fixture, value = worker_hardlink_fixture(tmp_path)
    deps = Path(value["roots"][1])
    alias = Path(value["paths"][1])
    real_deps = deps.with_name("real-deps")
    deps.rename(real_deps)
    deps.symlink_to(real_deps, target_is_directory=True)
    assert (real_deps / alias.name).exists()
    with pytest.raises(BUNDLE.BundleError, match="symlink"):
        validate_worker_fixture(fixture)


@pytest.fixture(scope="module")
def trusted_reconstruction():
    return BUNDLE.reconstruct()


def rebind_transport(root: Path) -> None:
    bundle_path = root / "bundle.json"
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    for name in BUNDLE.REQUIRED_FILES:
        if name in bundle["files"]:
            bundle["files"][name]["sha256"] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    rewrite_json(bundle_path, bundle)
    lines = []
    for name in sorted([*BUNDLE.REQUIRED_FILES, "bundle.json"]):
        lines.append(f"{hashlib.sha256((root / name).read_bytes()).hexdigest()}  {name}\n")
    sums = root / "SHA256SUMS"
    sums.chmod(0o644)
    sums.write_text("".join(lines), encoding="ascii")
    sums.chmod(0o444)


def test_checked_in_bundle_passes_offline_validation(trusted_reconstruction) -> None:
    value = BUNDLE.validate(ARTIFACT, trusted_reconstruction)
    assert value["status"] == "prepared_not_executed"
    assert value["promotion"] is False
    assert value["offline_evidence"] == {
        "trust_root_reconstruction": "passed",
        "schema_hash_path_link_toctou_validation": "passed",
        "trusted_runner_subprocess_required": True,
        "runner_dry_run": "passed",
        "synthetic_fake_ready_validation": "passed",
        "model_load_executed": False,
        "gpu_command_executed": False,
        "service_touched": False,
    }
    assert value["actual_live_observations"]["runtime_identity"] is None
    assert value["actual_live_observations"]["power"] is None
    assert value["actual_live_observations"]["vram"] is None
    assert value["resident_driver"]["source_commit"] == BUNDLE.DRIVER_COMMIT
    assert value["resident_driver"]["source_tree"] == BUNDLE.DRIVER_TREE
    assert value["resident_driver"]["source_git_blob"] == BUNDLE.DRIVER_SOURCE_GIT_BLOB
    assert value["resident_driver"]["source_sha256"] == BUNDLE.DRIVER_SOURCE_SHA
    assert value["resident_driver"]["binary_sha256"] == BUNDLE.EXPECTED_DRIVER_SHA
    assert value["resident_driver"]["binary_bytes"] == BUNDLE.EXPECTED_DRIVER_BYTES
    assert value["resident_driver"]["binary_build_id_sha1"] == BUNDLE.EXPECTED_DRIVER_BUILD_ID
    assert value["resident_driver"]["build"] == BUNDLE.DRIVER_BUILD_METADATA
    launch = json.loads((ARTIFACT / "launch-command.json").read_text())
    assert launch["bindings"]["driver"] == {
        "path": str(ARTIFACT / "resident-driver"),
        "sha256": BUNDLE.EXPECTED_DRIVER_SHA,
        "source_commit": BUNDLE.DRIVER_COMMIT,
        "source_tree": BUNDLE.DRIVER_TREE,
        "source_git_blob": BUNDLE.DRIVER_SOURCE_GIT_BLOB,
        "source_sha256": BUNDLE.DRIVER_SOURCE_SHA,
        "build": BUNDLE.DRIVER_BUILD_METADATA,
    }
    assert value["runner"]["source_commit"] == BUNDLE.RUNNER_COMMIT
    assert value["historical_predecessor"] == {
        "source_commit": "0fd7993843d0d7f1096d89079ce06922871d9f1a",
        "status": "superseded_historical_prepared",
        "execution_eligible": False,
    }
    superseded = json.loads((ARTIFACT / "SUPERSEDED-0fd7993.json").read_text())
    assert superseded["execution_eligible"] is False
    assert superseded["promotion"] is False


def test_fd_map_validate_uses_logical_paths_and_pread_without_moving_offsets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    trusted_reconstruction,
) -> None:
    root = copy_bundle(tmp_path)
    with inherited_bundle_map(root) as (descriptor, value):
        control_descriptor = value["bindings"][1]["descriptor"]
        os.lseek(descriptor, 7, os.SEEK_SET)
        os.lseek(control_descriptor, 3, os.SEEK_SET)
        before = (os.lseek(descriptor, 0, os.SEEK_CUR), os.lseek(control_descriptor, 0, os.SEEK_CUR))
        monkeypatch.setenv(BUNDLE.FD_MAP_ENV, str(descriptor))
        pinned = BUNDLE.PinnedFdMap.from_environment()
        assert pinned is not None
        result = BUNDLE.validate(root, trusted_reconstruction, pinned)
        after = (os.lseek(descriptor, 0, os.SEEK_CUR), os.lseek(control_descriptor, 0, os.SEEK_CUR))
    assert result["status"] == "prepared_not_executed"
    assert before == after
    assert result["canonical_root"] == str(BUNDLE.CANONICAL_ROOT)


def test_fd_map_control_path_swap_reads_only_the_pinned_old_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = copy_bundle(tmp_path)
    control = (tmp_path / "external-control.json").resolve()
    control.write_bytes(b'{"trusted":true}\n')
    with inherited_bundle_map(root, extra_control=control) as (descriptor, _value):
        monkeypatch.setenv(BUNDLE.FD_MAP_ENV, str(descriptor))
        pinned = BUNDLE.PinnedFdMap.from_environment()
        assert pinned is not None
        item = pinned.binding(control)
        assert item is not None
        old = tmp_path / "old-external-control.json"
        control.rename(old)
        control.write_bytes(b'{"trusted":false}\n')
        assert fd_identity(item["descriptor"])["ctime_ns"] != item["identity"]["ctime_ns"]
        raw, digest, _metadata = pinned.read(item)
    assert raw == b'{"trusted":true}\n'
    assert digest == hashlib.sha256(raw).hexdigest()


def test_fd_map_bundle_root_logical_swap_fails_the_seven_field_guard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = copy_bundle(tmp_path)
    with inherited_bundle_map(root) as (descriptor, _value):
        monkeypatch.setenv(BUNDLE.FD_MAP_ENV, str(descriptor))
        pinned = BUNDLE.PinnedFdMap.from_environment()
        assert pinned is not None
        old_root = tmp_path / "old-bundle"
        root.rename(old_root)
        shutil.copytree(ARTIFACT, root)
        with pytest.raises(BUNDLE.BundleError, match="bundle[_ ]root|guarded data path"):
            BUNDLE.validate(root, pinned_map=pinned)


def test_fd_map_missing_bundle_control_binding_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    trusted_reconstruction,
) -> None:
    root = copy_bundle(tmp_path)

    def remove_policy(value: dict) -> None:
        value["bindings"] = [
            item for item in value["bindings"] if item["logical_path"] != str(root / "policy.json")
        ]

    with inherited_bundle_map(root, mutate=remove_policy) as (descriptor, _value):
        monkeypatch.setenv(BUNDLE.FD_MAP_ENV, str(descriptor))
        pinned = BUNDLE.PinnedFdMap.from_environment()
        assert pinned is not None
        with pytest.raises(BUNDLE.BundleError, match="absent from pinned FD map"):
            BUNDLE.validate(root, trusted_reconstruction, pinned)


@pytest.mark.parametrize(
    ("variant", "message"),
    (
        ("root_extra", "root fields"),
        ("binding_extra", "binding fields"),
        ("duplicate_role", "binding value"),
        ("identity", "identity/type"),
        ("resolved_path", "binding value"),
        ("null_control_sha", "binding value"),
    ),
)
def test_fd_map_exact_schema_and_identity_reject_mutations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
    message: str,
) -> None:
    root = copy_bundle(tmp_path)

    def mutate(value: dict) -> None:
        if variant == "root_extra":
            value["unexpected"] = False
        elif variant == "binding_extra":
            value["bindings"][1]["unexpected"] = False
        elif variant == "duplicate_role":
            value["bindings"][1]["role"] = value["bindings"][0]["role"]
        elif variant == "identity":
            value["bindings"][1]["identity"]["inode"] += 1
        elif variant == "resolved_path":
            value["bindings"][1]["resolved_path"] = str(root / "policy.json")
        else:
            value["bindings"][1]["sha256"] = None

    with inherited_bundle_map(root, mutate=mutate) as (descriptor, _value):
        monkeypatch.setenv(BUNDLE.FD_MAP_ENV, str(descriptor))
        with pytest.raises(BUNDLE.BundleError, match=message):
            BUNDLE.PinnedFdMap.from_environment()


@pytest.mark.parametrize(("sealed", "corrupt_hash", "message"), ((False, False, "seals"), (True, True, "self-hash")))
def test_fd_map_rejects_unsealed_or_self_hash_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sealed: bool,
    corrupt_hash: bool,
    message: str,
) -> None:
    root = copy_bundle(tmp_path)
    with inherited_bundle_map(root, sealed=sealed, corrupt_hash=corrupt_hash) as (descriptor, _value):
        monkeypatch.setenv(BUNDLE.FD_MAP_ENV, str(descriptor))
        with pytest.raises(BUNDLE.BundleError, match=message):
            BUNDLE.PinnedFdMap.from_environment()


def test_rejects_unknown_bundle_schema(tmp_path: Path, trusted_reconstruction) -> None:
    root = copy_bundle(tmp_path)
    path = root / "bundle.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value["schema_version"] = "ullm.aq4_p2_resident_smoke_binding_bundle.v999"
    rewrite_json(path, value)
    rebind_transport(root)
    with pytest.raises(BUNDLE.BundleError, match="semantic reconstruction differs: bundle"):
        BUNDLE.validate(root, trusted_reconstruction)


def test_rejects_payload_hash_mutation(tmp_path: Path, trusted_reconstruction) -> None:
    root = copy_bundle(tmp_path)
    path = root / "fixture.json"
    path.chmod(0o644)
    path.write_bytes(path.read_bytes() + b" ")
    path.chmod(0o444)
    rebind_transport(root)
    with pytest.raises(BUNDLE.BundleError, match="semantic reconstruction differs: fixture"):
        BUNDLE.validate(root, trusted_reconstruction)


def test_rejects_unsafe_member_path(tmp_path: Path) -> None:
    with pytest.raises(BUNDLE.BundleError, match="unsafe bundle member path"):
        BUNDLE._safe_member(tmp_path, "../fixture.json")
    with pytest.raises(BUNDLE.BundleError, match="unsafe bundle member path"):
        BUNDLE._safe_member(tmp_path, "/tmp/fixture.json")


def test_rejects_symlink_member(tmp_path: Path, trusted_reconstruction) -> None:
    root = copy_bundle(tmp_path)
    fixture = root / "fixture.json"
    target = tmp_path / "external-fixture.json"
    shutil.copyfile(fixture, target)
    fixture.unlink()
    fixture.symlink_to(target)
    with pytest.raises(BUNDLE.BundleError, match="type/link/mode"):
        BUNDLE.validate(root, trusted_reconstruction)


def test_rejects_hardlink_member(tmp_path: Path, trusted_reconstruction) -> None:
    root = copy_bundle(tmp_path)
    fixture = root / "fixture.json"
    target = tmp_path / "external-fixture.json"
    shutil.copyfile(fixture, target)
    target.chmod(0o444)
    fixture.unlink()
    os.link(target, fixture)
    with pytest.raises(BUNDLE.BundleError, match="type/link/mode"):
        BUNDLE.validate(root, trusted_reconstruction)


def test_final_pass_detects_toctou_mutation(tmp_path: Path, trusted_reconstruction) -> None:
    root = copy_bundle(tmp_path)

    def mutate(bundle_root: Path) -> None:
        path = bundle_root / "policy.json"
        path.chmod(0o644)
        path.write_bytes(path.read_bytes() + b" ")
        path.chmod(0o444)

    BUNDLE._VALIDATION_HOOK = mutate
    try:
        with pytest.raises(BUNDLE.BundleError, match="TOCTOU mutation detected"):
            BUNDLE.validate(root, trusted_reconstruction)
    finally:
        BUNDLE._VALIDATION_HOOK = None


def test_package_tree_hash_matches_driver_algorithm_and_rejects_symlink(tmp_path: Path) -> None:
    package = tmp_path / "package"
    package.mkdir()
    (package / "a.bin").write_bytes(b"a")
    nested = package / "nested"
    nested.mkdir()
    (nested / "b.bin").write_bytes(b"b")
    aggregate = __import__("hashlib").sha256()
    for relative in ("a.bin", "nested/b.bin"):
        digest = __import__("hashlib").sha256((package / relative).read_bytes()).digest()
        aggregate.update(relative.encode() + b"\0" + digest + b"\n")
    actual, count = BUNDLE.package_tree_sha256(package)
    assert actual == aggregate.hexdigest()
    assert count == 2
    (package / "link").symlink_to(package / "a.bin")
    with pytest.raises(BUNDLE.BundleError, match="symlink rejected"):
        BUNDLE.package_tree_sha256(package)


@pytest.mark.parametrize(
    "variant",
    ("official_case", "runtime_case", "fixture", "identity", "preflight", "policy", "served_model", "trust_roots"),
)
def test_semantic_rebind_is_rejected_from_independent_trust_roots(tmp_path: Path, trusted_reconstruction, variant: str) -> None:
    root = copy_bundle(tmp_path)
    if variant == "official_case":
        path = root / "official-case.json"; value = json.loads(path.read_text()); value["case"]["prompt_tokens"] = 129
    elif variant == "runtime_case":
        path = root / "case-binding.json"; value = json.loads(path.read_text()); value["cases"][0]["device"]["runtime_device_index"] = 0; value["runtime_binding"]["bound_device"]["runtime_device_index"] = 0
    elif variant == "fixture":
        path = root / "fixture.json"; value = json.loads(path.read_text()); value["cases"][0]["prompt_token_ids"][0] += 1
    elif variant == "identity":
        path = root / "identity.json"; value = json.loads(path.read_text()); value["resident_driver_identity"]["model_revision"] = "semantic-rebind"
    elif variant == "preflight":
        path = root / "preflight.json"; value = json.loads(path.read_text()); value["workspace_bytes"] = 1
    elif variant == "policy":
        path = root / "policy.json"; value = json.loads(path.read_text()); value["status"] = "semantic-rebind"
    elif variant == "served_model":
        path = root / "served-model.json"; value = json.loads(path.read_text()); value["public"]["revision"] = "semantic-rebind"
    else:
        path = root / "trust-roots.json"; value = json.loads(path.read_text()); value["source"]["tree"] = "0" * 40
    rewrite_json(path, value)
    rebind_transport(root)
    with pytest.raises(BUNDLE.BundleError, match="independent semantic reconstruction differs"):
        BUNDLE.validate(root, trusted_reconstruction)


def test_nested_unknown_and_duplicate_fields_are_rejected(tmp_path: Path, trusted_reconstruction) -> None:
    unknown_root = copy_bundle(tmp_path / "unknown")
    policy = unknown_root / "policy.json"
    value = json.loads(policy.read_text()); value["nested_unknown"] = {"accepted": False}
    rewrite_json(policy, value); rebind_transport(unknown_root)
    with pytest.raises(BUNDLE.BundleError, match="semantic reconstruction differs"):
        BUNDLE.validate(unknown_root, trusted_reconstruction)

    duplicate_root = copy_bundle(tmp_path / "duplicate")
    identity = duplicate_root / "identity.json"
    raw = identity.read_text(encoding="utf-8").replace('  "status": "bound"', '  "status": "bound",\n  "status": "bound"', 1)
    identity.chmod(0o644); identity.write_text(raw, encoding="utf-8"); identity.chmod(0o444)
    rebind_transport(duplicate_root)
    with pytest.raises(BUNDLE.BundleError, match="duplicate JSON key"):
        BUNDLE.validate(duplicate_root, trusted_reconstruction)


@pytest.mark.parametrize("variant", ("late_unknown", "late_missing", "late_replace"))
def test_final_directory_reenumeration_rejects_late_mutation(tmp_path: Path, trusted_reconstruction, variant: str) -> None:
    root = copy_bundle(tmp_path)
    replacement = tmp_path / "replacement-policy.json"
    shutil.copyfile(root / "policy.json", replacement)
    replacement.chmod(0o444)

    def mutate(bundle_root: Path) -> None:
        if variant == "late_unknown":
            (bundle_root / "late-unknown.json").write_text("{}\n", encoding="utf-8")
        elif variant == "late_missing":
            (bundle_root / "policy.json").unlink()
        else:
            (bundle_root / "policy.json").unlink()
            shutil.copyfile(replacement, bundle_root / "policy.json")
            (bundle_root / "policy.json").chmod(0o444)

    BUNDLE._VALIDATION_HOOK = mutate
    try:
        with pytest.raises(BUNDLE.BundleError, match="late bundle directory mutation"):
            BUNDLE.validate(root, trusted_reconstruction)
    finally:
        BUNDLE._VALIDATION_HOOK = None


@pytest.mark.parametrize("variant", ("relative_served", "parent_driver", "served_sha", "extra_arg"))
def test_launch_command_rejects_nonexact_following_arguments_and_bindings(tmp_path: Path, variant: str) -> None:
    value = json.loads((ARTIFACT / "launch-command.json").read_text())
    if variant == "relative_served":
        value["resident_driver_argv"][2] = "active.json"
    elif variant == "parent_driver":
        value["resident_driver_argv"][0] = str(BUNDLE.CANONICAL_ROOT / "subdir/../resident-driver")
    elif variant == "served_sha":
        value["bindings"]["served_model_manifest"]["sha256"] = "0" * 64
    else:
        value["resident_driver_argv"].append("--unexpected")
    with pytest.raises(BUNDLE.BundleError, match="launch command"):
        BUNDLE.validate_launch_command(value)


def test_launch_path_rejects_parent_symlink(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    (real / "driver").write_bytes(b"driver")
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    with pytest.raises(BUNDLE.BundleError, match="symlink component"):
        BUNDLE.reject_symlink_components(alias / "driver", "launch driver")


def test_runner_generated_plan_and_subprocess_evidence_are_bound() -> None:
    plan_raw = (ARTIFACT / "dry-run.json").read_bytes()
    plan = json.loads(plan_raw)
    evidence = json.loads((ARTIFACT / "runner-dry-run-evidence.json").read_text())
    assert (plan["case_count"], plan["transaction_count"], plan["warmup_runs"], plan["measured_runs"]) == (1, 12, 2, 10)
    assert plan["execution_mode"] == "one_case_smoke"
    assert plan["smoke_only"] is True
    assert plan["promotion_eligible"] is False
    assert plan["validation"]["mode"] == "validate_only"
    assert plan["validation"]["driver_fake_handshake"] == "passed"
    assert evidence["runner_subprocess_count"] == 1
    assert evidence["exit_code"] == 0
    assert evidence["stdout"] == {"sha256": hashlib.sha256(b"").hexdigest(), "utf8": ""}
    assert evidence["stderr"] == {"sha256": hashlib.sha256(b"").hexdigest(), "utf8": ""}
    assert evidence["plan"]["sha256"] == hashlib.sha256(plan_raw).hexdigest()
    assert evidence["normal_profile"] == {"case_count": 84, "separate": True}


@pytest.mark.parametrize("variant", ("smoke_only", "transaction_count", "normal_profile"))
def test_rejects_rebound_runner_plan_or_normal_profile(tmp_path: Path, trusted_reconstruction, variant: str) -> None:
    root = copy_bundle(tmp_path)
    if variant == "normal_profile":
        path = root / "runner-dry-run-evidence.json"
        value = json.loads(path.read_text())
        value["normal_profile"]["case_count"] = 1
    else:
        path = root / "dry-run.json"
        value = json.loads(path.read_text())
        value[variant] = False if variant == "smoke_only" else 84
    rewrite_json(path, value)
    rebind_transport(root)
    with pytest.raises(BUNDLE.BundleError, match="trusted runner"):
        BUNDLE.validate(root, trusted_reconstruction)


def test_binding_actual_runner_authority_matches_76c48aa_source() -> None:
    commit = BUNDLE.BINDING_SOURCE_COMMIT
    source_path = "tools/run-aq4-p2-resident-batch.py"
    assert commit == "76c48aa27c08f8cd5115a15e6be25b83d679d8fa"
    assert subprocess.run(
        ["git", "rev-parse", f"{commit}^{{tree}}"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip() == BUNDLE.BINDING_SOURCE_TREE
    assert subprocess.run(
        ["git", "rev-parse", f"{commit}:{source_path}"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip() == BUNDLE.BINDING_RUNNER_GIT_BLOB
    source = subprocess.run(
        ["git", "show", f"{commit}:{source_path}"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        check=True,
    ).stdout
    assert hashlib.sha256(source).hexdigest() == BUNDLE.BINDING_RUNNER_SHA
    assert BUNDLE.BINDING_RUNNER_SHA == (
        "bbe978ede0e4662c33d0d12eee4194531f340b9c06001f37d619019197fd5138"
    )


def test_checked_in_v6_binding_sidecar_passes_and_pins_final_runner_validator() -> None:
    value = BUNDLE.validate_binding(VALIDATOR_COMMIT, VALIDATOR_SHA, BINDING)
    assert stat.S_IMODE(BINDING.lstat().st_mode) == BUNDLE.BINDING_ROOT_MODE == 0o555
    assert value["schema_version"] == "ullm.aq4_p2_resident_smoke_binding.v6"
    assert value["status"] == "prepared_not_executed"
    assert value["promotion"] is False
    assert value["launch_eligible"] is False
    assert value["requires_immutable_launcher"] is True
    assert value["predecessor"] == {"commit": "791a20c", "status": "SUPERSEDED", "execution_eligible": False}
    roots = value["trust_roots"]
    assert roots["source_commit"] == BUNDLE.BINDING_SOURCE_COMMIT
    assert roots["source_tree"] == BUNDLE.BINDING_SOURCE_TREE
    assert roots["runner"] == {
        "source_commit": BUNDLE.BINDING_SOURCE_COMMIT,
        "source_tree": BUNDLE.BINDING_SOURCE_TREE,
        "git_blob": BUNDLE.BINDING_RUNNER_GIT_BLOB,
        "source_sha256": BUNDLE.BINDING_RUNNER_SHA,
        "archive_path": str(BINDING / "trusted-runner.py"),
    }
    assert roots["validator"]["source_commit"] == VALIDATOR_COMMIT
    assert roots["validator"]["sha256"] == VALIDATOR_SHA
    assert roots["resident_driver"]["blob_unchanged"] is True
    assert roots["resident_driver"]["source_tree"] == BUNDLE.DRIVER_TREE
    assert roots["resident_driver"]["git_blob_at_binding_commit"] == BUNDLE.DRIVER_SOURCE_GIT_BLOB
    assert roots["resident_driver"]["source_sha256"] == BUNDLE.DRIVER_SOURCE_SHA
    assert roots["resident_driver"]["binary_sha256"] == BUNDLE.EXPECTED_DRIVER_SHA
    assert roots["resident_driver"]["binary_bytes"] == BUNDLE.EXPECTED_DRIVER_BYTES
    assert roots["resident_driver"]["binary_build_id_sha1"] == BUNDLE.EXPECTED_DRIVER_BUILD_ID
    assert roots["resident_driver"]["build"] == BUNDLE.DRIVER_BUILD_METADATA


def test_v6_binding_runner_and_validator_archives_match_pinned_git_objects() -> None:
    manifest = json.loads((BINDING / "binding-manifest.json").read_text())
    for role, source_path, archive_name in (
        ("runner", "tools/run-aq4-p2-resident-batch.py", "trusted-runner.py"),
        ("validator", "tools/prepare-aq4-p2-resident-smoke-bundle.py", "trusted-validator.py"),
    ):
        authority = manifest["trust_roots"][role]
        commit = authority["source_commit"]
        observed_tree = subprocess.run(
            ["git", "rev-parse", f"{commit}^{{tree}}"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        observed_blob = subprocess.run(
            ["git", "rev-parse", f"{commit}:{source_path}"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        archived = (BINDING / archive_name).read_bytes()
        committed = subprocess.run(
            ["git", "show", f"{commit}:{source_path}"],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            check=True,
        ).stdout
        assert observed_tree == authority["source_tree"]
        assert observed_blob == authority["git_blob"]
        assert archived == committed
        assert hashlib.sha256(archived).hexdigest() == authority["source_sha256" if role == "runner" else "sha256"]


def test_v6_binding_records_actual_runner_and_mandatory_validator_subprocesses() -> None:
    plan_raw = (BINDING / "runner-plan.json").read_bytes()
    plan = json.loads(plan_raw)
    evidence = json.loads((BINDING / "runner-subprocess-evidence.json").read_text())
    report_raw = (BINDING / "validator-report.json").read_bytes()
    validator = plan["validation"]["trusted_bundle_validator"]
    assert (plan["case_count"], plan["transaction_count"], plan["warmup_runs"], plan["measured_runs"]) == (1, 12, 2, 10)
    assert plan["smoke_only"] is True and plan["promotion_eligible"] is False
    assert plan["validation"]["root_contract"] == "ullm.aq4_p2_resident_smoke_bundle_root.v4"
    assert set(plan["validation"]["members"]) == set(BUNDLE.REQUIRED_FILES) | {"bundle.json", "SHA256SUMS"}
    assert plan["validation"]["fake_driver_subprocess_count"] == 1
    assert plan["validation"]["fake_ready_scope"] == BUNDLE.BINDING_FAKE_READY_SCOPE == {
        "stage": "pre_spawn_fixture_only",
        "runtime_proof": False,
        "ready_proof": False,
        "model_load_proof": False,
    }
    assert plan["validation"]["resident_driver_argv"] == BUNDLE.resident_driver_argv()
    assert validator["subprocess_count"] == 1
    assert validator["source"] == {"path": str(BUNDLE.BINDING_VALIDATOR_EXEC), "sha256": VALIDATOR_SHA}
    assert validator["report_sha256"] == hashlib.sha256(BUNDLE.canonical(validator["report"])).hexdigest()
    assert evidence["runner_subprocess_count"] == 1
    assert evidence["runner_source"] == {
        "source_commit": BUNDLE.BINDING_SOURCE_COMMIT,
        "source_tree": BUNDLE.BINDING_SOURCE_TREE,
        "git_blob": BUNDLE.BINDING_RUNNER_GIT_BLOB,
        "source_sha256": BUNDLE.BINDING_RUNNER_SHA,
        "archive_path": str(BINDING / "trusted-runner.py"),
    }
    assert evidence["exit_code"] == 0
    assert evidence["stdout"] == {"sha256": hashlib.sha256(b"").hexdigest(), "utf8": ""}
    assert evidence["stderr"] == {"sha256": hashlib.sha256(b"").hexdigest(), "utf8": ""}
    assert evidence["plan"]["sha256"] == hashlib.sha256(plan_raw).hexdigest()
    assert evidence["trusted_validator"]["report_file_sha256"] == hashlib.sha256(report_raw).hexdigest()


def test_v6_binding_keeps_generic_runner_outputs_outside_immutable_input_root() -> None:
    manifest = json.loads((BINDING / "binding-manifest.json").read_text())
    assert manifest["binding_root_contract"] == {
        "type": "directory",
        "mode": "0555",
        "members_single_link": True,
        "members_read_only": True,
    }
    assert manifest["runner_roles"] == {
        "prepared_bootstrap": {
            "commit": BUNDLE.RUNNER_COMMIT,
            "sha256": BUNDLE.RUNNER_SOURCE_SHA,
            "role": "historical_control_member",
            "execution_closure": "control_input/read",
        },
        "binding_actual": {
            "commit": BUNDLE.BINDING_SOURCE_COMMIT,
            "sha256": BUNDLE.BINDING_RUNNER_SHA,
            "role": "actual_generic_runner",
            "execution_closure": "code_execution/exec",
        },
        "same_runner": False,
    }
    assert manifest["cycle_control"] == {
        "input_root_unchanged_after_runner": True,
        "generated_outputs_outside_input_root": True,
        "input_root_dry_run_not_replaced": True,
        "generic_runner_schema_not_embedded_back_into_input_root": True,
    }
    assert manifest["input_root"]["members"]["dry-run.json"]["sha256"] == hashlib.sha256((ARTIFACT / "dry-run.json").read_bytes()).hexdigest()
    assert manifest["outputs"]["runner_plan_sha256"] == hashlib.sha256((BINDING / "runner-plan.json").read_bytes()).hexdigest()
    assert manifest["next_stage"]["name"] == "L immutable launcher"
    assert manifest["next_stage"]["required"] is True


def test_v6_binding_rejects_report_replacement(tmp_path: Path) -> None:
    root = tmp_path / "binding"
    shutil.copytree(BINDING, root)
    report = root / "validator-report.json"
    report.chmod(0o644)
    value = json.loads(report.read_text())
    value["promotion"] = True
    rewrite_json(report, value)
    with pytest.raises(BUNDLE.BundleError, match="validator report differs"):
        BUNDLE.validate_binding(VALIDATOR_COMMIT, VALIDATOR_SHA, root)


def test_v6_binding_rejects_writable_root(tmp_path: Path) -> None:
    root = tmp_path / "binding"
    shutil.copytree(BINDING, root)
    root.chmod(0o775)
    with pytest.raises(BUNDLE.BundleError, match="binding root mode differs"):
        BUNDLE.validate_binding(VALIDATOR_COMMIT, VALIDATOR_SHA, root)


@pytest.mark.parametrize("variant", ("late_unknown", "late_missing", "late_replace"))
def test_v6_binding_final_reenumeration_rejects_late_mutation(tmp_path: Path, variant: str) -> None:
    root = tmp_path / "binding"
    shutil.copytree(BINDING, root)
    replacement = tmp_path / "replacement-report.json"
    shutil.copyfile(root / "validator-report.json", replacement)
    replacement.chmod(0o444)

    def mutate(binding_root: Path) -> None:
        binding_root.chmod(0o755)
        if variant == "late_unknown":
            (binding_root / "late-unknown.json").write_text("{}\n", encoding="utf-8")
        elif variant == "late_missing":
            (binding_root / "validator-report.json").unlink()
        else:
            (binding_root / "validator-report.json").unlink()
            shutil.copyfile(replacement, binding_root / "validator-report.json")
            (binding_root / "validator-report.json").chmod(0o444)

    BUNDLE._BINDING_VALIDATION_HOOK = mutate
    try:
        with pytest.raises(BUNDLE.BundleError, match="late binding sidecar"):
            BUNDLE.validate_binding(VALIDATOR_COMMIT, VALIDATOR_SHA, root)
    finally:
        BUNDLE._BINDING_VALIDATION_HOOK = None
