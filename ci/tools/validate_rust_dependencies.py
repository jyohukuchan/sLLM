#!/usr/bin/env python3
"""Validate the checked-in, offline Rust dependency closure.

The checked-in policy deliberately contains normalized identities instead of
Cargo's path-bearing package IDs.  The normalizer is the only place that
looks at Cargo metadata IDs and cached package paths; those values never cross
the policy boundary.
"""

from __future__ import annotations

import argparse
import copy
import os
import platform
import re
import subprocess
import sys
import tomllib
from pathlib import Path
from typing import Any, Callable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import ContractError, ROOT, read_json  # noqa: E402

POLICY_PATH = Path("ci/dependencies/rust-workspace-v1.json")
SCHEMA_PATH = Path("ci/schema/rust-dependency-policy-v1.schema.json")
REGISTRY_SOURCE = "registry+https://github.com/rust-lang/crates.io-index"
EXPECTED_SCHEMA_VERSION = "rust-dependency-policy-v1"
EXPECTED_POLICY_ID = "rust-workspace-v1"
MSRV_AUTHORITY = "1.85.0"
MSRV_TARGET = "x86_64-unknown-linux-gnu"
ALLOWED_EDGE_KINDS = {"normal", "build", "dev"}
PACKAGE_SOURCES = {REGISTRY_SOURCE, "workspace"}
TOKENIZERS_ALLOWED_FEATURES = ["onig"]
TOKENIZERS_FORBIDDEN_FEATURES = ["default", "http", "progressbar", "esaxx_fast"]
TOKENIZERS_PACKAGE = "registry:tokenizers@0.21.4"
ESAXX_PACKAGE = "registry:esaxx-rs@0.1.10"
MINIJINJA_PACKAGE = "registry:minijinja@2.24.0"
MINIJINJA_REQUIRED_PACKAGES = ["registry:memo-map@0.3.3"]
MINIJINJA_REQUESTED_FEATURES = [
    "builtins", "fuel", "json", "macros", "multi_template", "serde",
]
MINIJINJA_RESOLVED_FEATURES = [
    "builtins", "fuel", "json", "macros", "multi_template", "serde", "serde_json",
]
MINIJINJA_FORBIDDEN_FEATURES = [
    "custom_syntax", "debug", "default", "deserialization", "loader", "stacker", "urlencode",
]
WASIP2_PACKAGE = "registry:wasip2@1.0.4+wasi-0.2.12"
WASIP2_TARGET = 'cfg(all(target_arch = "wasm32", target_os = "wasi", target_env = "p2"))'
EXPECTED_PACKAGE_COUNT = 190
EXPECTED_REGISTRY_PACKAGE_COUNT = 184
EXPECTED_WORKSPACE_PACKAGE_COUNT = 6
EXPECTED_EDGE_COUNT = 447
SERVER_PACKAGE = "workspace:sllm-server@0.1.0"
SERVER_RUNTIME_DEPENDENCIES = [
    {
        "package": "registry:axum-server@0.8.0",
        "requested": ["tls-rustls"],
        "resolved": [
            "arc-swap", "rustls", "rustls-pki-types", "tls-rustls",
            "tls-rustls-no-provider", "tokio-rustls",
        ],
        "uses_default_features": False,
    },
    {
        "package": "registry:axum@0.8.9",
        "requested": ["http1", "json", "tokio"],
        "resolved": ["http1", "json", "tokio"],
        "uses_default_features": False,
    },
    {
        "package": "registry:futures-util@0.3.33",
        "requested": ["std"],
        "resolved": ["alloc", "slab", "std"],
        "uses_default_features": False,
    },
    {
        "package": "registry:serde_path_to_error@0.1.20",
        "requested": [],
        "resolved": [],
        "uses_default_features": True,
    },
    {
        "package": "registry:tokio-stream@0.1.19",
        "requested": ["sync"],
        "resolved": ["sync", "tokio-util"],
        "uses_default_features": False,
    },
    {
        "package": "registry:tokio@1.53.1",
        "requested": ["macros", "net", "rt-multi-thread", "signal", "sync", "time"],
        "resolved": [
            "bytes", "default", "fs", "io-util", "libc", "macros", "mio", "net", "rt",
            "rt-multi-thread", "signal", "signal-hook-registry", "socket2", "sync", "time",
            "tokio-macros", "windows-sys",
        ],
        "uses_default_features": False,
    },
    {
        "package": "registry:tower-http@0.7.0",
        "requested": ["cors", "limit", "trace"],
        "resolved": ["cors", "limit", "trace", "tracing"],
        "uses_default_features": False,
    },
]
B0_DISABLED_HIP_FLAGS = frozenset({
    "SLLM_ENABLE_HIP_COMPILE_PROBE",
    "SLLM_ENABLE_HIP_RUNTIME",
    "SLLM_ENABLE_PUBLIC_HIP_RUNTIME",
})
B0_ABSENT_ENVIRONMENT_VARIABLES = frozenset({
    "CARGO_BUILD_TARGET",
    "CARGO_BUILD_RUSTC",
    "CARGO_BUILD_RUSTC_WRAPPER",
    "CARGO_BUILD_RUSTFLAGS",
    "CARGO_BUILD_RUSTDOCFLAGS",
    "CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUSTFLAGS",
    "CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUSTDOCFLAGS",
    "CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_LINKER",
    "CARGO_TARGET_X86_64_UNKNOWN_LINUX_GNU_RUNNER",
    "ROCM_PATH",
    "HIP_PATH",
    "CMAKE_HIP_ARCHITECTURES",
    "SLLM_HIP_CODEGEN_FEATURES",
    "SLLM_SEMANTIC_G1_AUTHORITY",
    "SLLM_HIP_COMPILER",
    "SLLM_HIP_COMPILER_LOGICAL",
    "SLLM_HIP_COMPILER_BROKER_SOCKET",
    "SLLM_HIP_COMPILER_BROKER_SESSION",
    "SLLM_HIP_COMPILER_BROKER_CLIENT",
    "SLLM_HIP_COMPILER_BROKER_CLIENT_SHA256",
    "SLLM_HIP_COMPILER_BROKER_CLIENT_FD",
    "SLLM_HIP_COMPILER_BROKER_TOKEN",
    "SLLM_SEMANTIC_G1_NATIVE_HIP_BUILD_DIR",
    "CXX",
    "CMAKE_HIP_COMPILER",
    "CMAKE_C_COMPILER",
    "CMAKE_CXX_COMPILER",
    "CMAKE_TOOLCHAIN_FILE",
    "CMAKE_PREFIX_PATH",
    "CMAKE_GENERATOR",
    "CMAKE_GENERATOR_PLATFORM",
    "CMAKE_GENERATOR_TOOLSET",
    "CMAKE_MAKE_PROGRAM",
    "CC",
    "HIPCC",
    "HIPCXX",
    "CFLAGS",
    "CXXFLAGS",
    "CPPFLAGS",
    "LDFLAGS",
    "CPATH",
    "C_INCLUDE_PATH",
    "CPLUS_INCLUDE_PATH",
    "OBJC_INCLUDE_PATH",
    "GCC_EXEC_PREFIX",
    "COMPILER_PATH",
    "LIBRARY_PATH",
    "LD_PRELOAD",
    "LD_LIBRARY_PATH",
    "RUSTC",
    "RUSTC_BOOTSTRAP",
    "RUSTC_WRAPPER",
    "RUSTC_WORKSPACE_WRAPPER",
    "RUSTFLAGS",
    "RUSTDOCFLAGS",
    "CARGO_ENCODED_RUSTFLAGS",
    "CARGO_ENCODED_RUSTDOCFLAGS",
})
B0_SANITIZED_ENVIRONMENT_VARIABLES = B0_DISABLED_HIP_FLAGS | B0_ABSENT_ENVIRONMENT_VARIABLES
SECTION_NAMES = ("workspace", "workspace_members", "packages", "edges", "counts")
VERSION_RE = re.compile(r"^(\d+)(?:\.(\d+))?(?:\.(\d+))?(?:[-+].*)?$")
IDENTITY_RE = re.compile(r"^(registry|workspace):[^:@\x00]+@[^\x00]+$")


def identity_key(name: str, version: str, source: str) -> str:
    """Return a stable policy identity with no filesystem component."""

    if source not in PACKAGE_SOURCES:
        raise ContractError(f"unknown normalized package source: {source!r}")
    return f"{'registry' if source == REGISTRY_SOURCE else 'workspace'}:{name}@{version}"


def package_identity(name: str, version: str, source: str) -> dict[str, str]:
    return {"name": name, "version": version, "source": source}


def package_identity_key(identity: dict[str, Any]) -> str:
    if set(identity) != {"name", "version", "source"}:
        raise ContractError(f"package identity keys are not closed: {identity!r}")
    name, version, source = identity["name"], identity["version"], identity["source"]
    if not all(isinstance(value, str) and value for value in (name, version, source)):
        raise ContractError(f"malformed package identity: {identity!r}")
    return identity_key(name, version, source)


def _relative_path(path: str, repo: Path) -> str:
    """Normalize an observed workspace manifest to a repo-relative path."""

    candidate = Path(path)
    try:
        relative = candidate.resolve().relative_to(repo.resolve())
    except (OSError, ValueError) as exc:
        raise ContractError(f"workspace manifest escapes repository: {path}") from exc
    result = relative.as_posix()
    if not result or result.startswith("/") or ".." in Path(result).parts:
        raise ContractError(f"unsafe workspace manifest path: {result!r}")
    return result


def _normalized_source(package: dict[str, Any], workspace_ids: set[str]) -> str:
    source = package.get("source")
    package_id = package.get("id")
    if source == REGISTRY_SOURCE:
        return REGISTRY_SOURCE
    if source is None and package_id in workspace_ids:
        return "workspace"
    raise ContractError(f"unknown or non-workspace package source: {package_id!r} / {source!r}")


def _target_record(target: dict[str, Any]) -> dict[str, Any]:
    allowed = {"name", "kind", "crate_types", "edition", "required_features", "test", "doc", "doctest"}
    if set(target) - {"kind", "crate_types", "edition", "name", "required-features", "test", "doc", "doctest", "src_path"}:
        raise ContractError(f"workspace target has unknown metadata keys: {target!r}")
    record = {
        "name": target["name"],
        "kind": list(target["kind"]),
        "crate_types": list(target["crate_types"]),
        "edition": target["edition"],
        "required_features": list(target.get("required-features", [])),
        "test": target["test"],
        "doc": target["doc"],
        "doctest": target["doctest"],
    }
    if set(record) != allowed:
        raise ContractError(f"workspace target normalization is incomplete: {target!r}")
    if any(not isinstance(value, bool) for value in (record["test"], record["doc"], record["doctest"])):
        raise ContractError(f"workspace target booleans are malformed: {target!r}")
    return record


def _rust_version_tuple(value: str) -> tuple[int, int, int]:
    match = VERSION_RE.fullmatch(value)
    if not match:
        raise ContractError(f"invalid declared rust-version: {value!r}")
    return tuple(int(part or 0) for part in match.groups())


def _edge_sort_key(edge: dict[str, Any]) -> tuple[str, str, str, str, str, str]:
    return (
        edge["from"], edge["to"], edge["name"], edge["kind"],
        "" if edge["target"] is None else edge["target"], edge["req"],
    )


def _cargo_dependency_name(name: str) -> str:
    """Cargo resolves a hyphenated package name through its underscore crate name."""

    return name.replace("-", "_")


def _dependency_lookup_name(dependency: dict[str, Any]) -> str | None:
    """Return the name Cargo uses in a resolve node for one manifest edge."""

    declared_name = dependency.get("rename")
    if declared_name is None:
        declared_name = dependency.get("name")
    if not isinstance(declared_name, str):
        return None
    return _cargo_dependency_name(declared_name)


def _find_declared_dependency(
    dependencies: list[dict[str, Any]],
    resolved_name: Any,
    *,
    kind: str,
    target: str | None,
) -> dict[str, Any] | None:
    """Map a resolve-node alias to exactly one manifest dependency."""

    candidates = [
        item for item in dependencies
        if _dependency_lookup_name(item) == resolved_name
        and (item.get("kind") or "normal") == kind
        and item.get("target") == target
    ]
    return candidates[0] if len(candidates) == 1 else None


def _package_sort_key(package: dict[str, Any]) -> str:
    return package_identity_key(package["identity"])


def _load_lock_packages(repo: Path) -> dict[str, dict[str, Any]]:
    try:
        with (repo / "Cargo.lock").open("rb") as stream:
            lock = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ContractError(f"cannot read Cargo.lock: {exc}") from exc
    packages = lock.get("package")
    if not isinstance(packages, list):
        raise ContractError("Cargo.lock has no package table")
    result: dict[str, dict[str, Any]] = {}
    for package in packages:
        if not isinstance(package, dict):
            raise ContractError("Cargo.lock contains a malformed package")
        source = package.get("source")
        if source is None:
            normalized = "workspace"
        elif source == REGISTRY_SOURCE:
            normalized = REGISTRY_SOURCE
        else:
            raise ContractError(f"Cargo.lock contains an unknown source: {source!r}")
        name, version = package.get("name"), package.get("version")
        if not isinstance(name, str) or not isinstance(version, str):
            raise ContractError(f"Cargo.lock package identity is malformed: {package!r}")
        key = identity_key(name, version, normalized)
        if key in result:
            raise ContractError(f"duplicate Cargo.lock package identity: {key}")
        if normalized == REGISTRY_SOURCE and not isinstance(package.get("checksum"), str):
            raise ContractError(f"registry Cargo.lock package has no checksum: {key}")
        result[key] = {"source": normalized, "checksum": package.get("checksum")}
    return result


def _root_workspace(repo: Path, metadata: dict[str, Any], workspace_ids: set[str]) -> tuple[dict[str, Any], dict[str, str]]:
    try:
        with (repo / "Cargo.toml").open("rb") as stream:
            root_manifest = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ContractError(f"cannot read root Cargo.toml: {exc}") from exc
    workspace = root_manifest.get("workspace")
    if not isinstance(workspace, dict) or workspace.get("resolver") != "3":
        raise ContractError("workspace resolver is not the locked resolver 3")
    members = workspace.get("members")
    if not isinstance(members, list) or not members or any(not isinstance(item, str) for item in members):
        raise ContractError("workspace members are malformed")
    member_by_id: dict[str, str] = {}
    for package in metadata["packages"]:
        if package["id"] not in workspace_ids:
            continue
        relative = _relative_path(package["manifest_path"], repo)
        member_by_id[package["id"]] = relative
    expected_members = sorted(
        (Path(member) / "Cargo.toml" if Path(member).name != "Cargo.toml" else Path(member)).as_posix()
        for member in members
    )
    if sorted(member_by_id.values()) != expected_members:
        raise ContractError(
            f"Cargo.toml workspace members differ from metadata: {expected_members!r} / {sorted(member_by_id.values())!r}"
        )
    default_ids = set(metadata.get("workspace_default_members", []))
    if default_ids != workspace_ids:
        raise ContractError("virtual workspace default members are incomplete")
    return (
        {
            "root": ".",
            "resolver": 3,
            "members": expected_members,
            "default_members": sorted(member_by_id[member_id] for member_id in default_ids),
        },
        member_by_id,
    )


def normalize_metadata(metadata: dict[str, Any], repo: Path = ROOT) -> dict[str, Any]:
    """Convert one Cargo metadata+lock observation to policy-shaped data."""

    packages = metadata.get("packages")
    resolve = metadata.get("resolve")
    workspace_members = metadata.get("workspace_members")
    if not isinstance(packages, list) or not isinstance(resolve, dict) or not isinstance(workspace_members, list):
        raise ContractError("Cargo metadata is missing packages, resolve, or workspace members")
    if resolve.get("root") is not None:
        raise ContractError("B0 requires a virtual workspace with resolve.root = null")
    workspace_ids = set(workspace_members)
    if len(workspace_ids) != len(workspace_members) or len(workspace_ids) != EXPECTED_WORKSPACE_PACKAGE_COUNT:
        raise ContractError("workspace member count or identity uniqueness drifted")
    workspace, manifest_by_id = _root_workspace(repo, metadata, workspace_ids)
    package_by_id = {package.get("id"): package for package in packages}
    if len(package_by_id) != len(packages) or any(not key for key in package_by_id):
        raise ContractError("metadata package IDs are missing or duplicated")
    if set(package_by_id) != {node.get("id") for node in resolve.get("nodes", [])}:
        raise ContractError("metadata package/node identity sets differ")

    normalized_source_by_id: dict[str, str] = {}
    identity_by_id: dict[str, str] = {}
    packages_out: list[dict[str, Any]] = []
    workspace_members_out: list[dict[str, Any]] = []
    for package in packages:
        package_id = package["id"]
        source = _normalized_source(package, workspace_ids)
        normalized_source_by_id[package_id] = source
        identity = package_identity(package["name"], package["version"], source)
        key = package_identity_key(identity)
        if key in identity_by_id.values():
            raise ContractError(f"duplicate normalized package identity: {key}")
        identity_by_id[package_id] = key
        node = next((item for item in resolve["nodes"] if item.get("id") == package_id), None)
        if not isinstance(node, dict):
            raise ContractError(f"missing resolve node: {package_id}")
        license_value = package.get("license")
        if source == REGISTRY_SOURCE and (not isinstance(license_value, str) or not license_value):
            raise ContractError(f"registry package has no license: {key}")
        rust_version = package.get("rust_version")
        if rust_version is not None and not isinstance(rust_version, str):
            raise ContractError(f"invalid rust-version for {key}: {rust_version!r}")
        package_out = {
            "identity": identity,
            "license": license_value,
            "rust_version": rust_version,
            "features": sorted(set(node.get("features", []))),
        }
        if source == "workspace":
            package_out["license"] = license_value
        packages_out.append(package_out)
        if package_id in workspace_ids:
            workspace_members_out.append(
                {
                    "identity": identity,
                    "manifest": manifest_by_id[package_id],
                    "targets": sorted(
                        (_target_record(target) for target in package.get("targets", [])),
                        key=lambda target: (target["name"], tuple(target["kind"])),
                    ),
                }
            )

    if len(packages_out) != EXPECTED_PACKAGE_COUNT:
        raise ContractError(f"resolved package count drifted: {len(packages_out)}")
    if sum(package["identity"]["source"] == REGISTRY_SOURCE for package in packages_out) != EXPECTED_REGISTRY_PACKAGE_COUNT:
        raise ContractError("resolved registry package count drifted")
    if len(workspace_members_out) != EXPECTED_WORKSPACE_PACKAGE_COUNT:
        raise ContractError("resolved workspace package count drifted")

    metadata_deps: dict[str, list[dict[str, Any]]] = {
        package_id: list(package.get("dependencies", [])) for package_id, package in package_by_id.items()
    }
    edges_out: list[dict[str, Any]] = []
    nodes = resolve.get("nodes")
    if not isinstance(nodes, list):
        raise ContractError("resolve.nodes is malformed")
    for node in nodes:
        from_id = node.get("id")
        if from_id not in identity_by_id:
            raise ContractError(f"resolve edge source is unknown: {from_id!r}")
        for dependency in node.get("deps", []):
            to_id = dependency.get("pkg")
            if to_id not in identity_by_id:
                raise ContractError(f"resolve edge destination is unknown: {to_id!r}")
            dep_kinds = dependency.get("dep_kinds")
            if not isinstance(dep_kinds, list) or not dep_kinds:
                raise ContractError("resolve dependency has no dep_kinds")
            for dep_kind in dep_kinds:
                kind = dep_kind.get("kind") or "normal"
                target = dep_kind.get("target")
                if kind not in ALLOWED_EDGE_KINDS or (target is not None and not isinstance(target, str)):
                    raise ContractError(f"invalid resolved edge kind/target: {dep_kind!r}")
                declared = _find_declared_dependency(
                    metadata_deps[from_id], dependency.get("name"), kind=kind, target=target
                )
                if declared is None:
                    raise ContractError(
                        f"resolved edge does not map to exactly one declared dependency: {from_id!r} -> {dependency!r}"
                    )
                declared_source = declared.get("source")
                expected_destination_source = normalized_source_by_id[to_id]
                if declared_source is None and expected_destination_source != "workspace":
                    raise ContractError("registry resolve edge has no declared registry source")
                if declared_source == REGISTRY_SOURCE and expected_destination_source != REGISTRY_SOURCE:
                    raise ContractError("workspace resolve edge is declared as registry")
                edges_out.append(
                    {
                        "from": identity_by_id[from_id],
                        "to": identity_by_id[to_id],
                        "name": declared["name"],
                        "kind": kind,
                        "target": target,
                        "req": declared["req"],
                        "requested_features": sorted(set(declared.get("features", []))),
                        "uses_default_features": declared["uses_default_features"],
                        "optional": declared["optional"],
                        "rename": declared.get("rename"),
                    }
                )
    if len(edges_out) != EXPECTED_EDGE_COUNT:
        raise ContractError(f"resolved edge count drifted: {len(edges_out)}")
    edge_keys = {
        (
            edge["from"], edge["to"], edge["name"], edge["kind"], edge["target"],
            edge["req"], tuple(edge["requested_features"]), edge["uses_default_features"],
            edge["optional"], edge["rename"],
        )
        for edge in edges_out
    }
    if len(edge_keys) != len(edges_out):
        raise ContractError("duplicate normalized resolve edge")

    lock_packages = _load_lock_packages(repo)
    observed_keys = set(identity_by_id.values())
    if set(lock_packages) != observed_keys:
        raise ContractError("Cargo.lock package identities differ from metadata")
    for package_out in packages_out:
        key = _package_sort_key(package_out)
        lock_entry = lock_packages[key]
        if package_out["identity"]["source"] == REGISTRY_SOURCE:
            checksum = lock_entry.get("checksum")
            if not isinstance(checksum, str):
                raise ContractError(f"registry package has no lock checksum: {key}")
            package_out["checksum"] = checksum
        else:
            if lock_entry.get("checksum") is not None:
                raise ContractError(f"workspace package unexpectedly has a checksum: {key}")
            package_out["checksum"] = None

    packages_out.sort(key=_package_sort_key)
    workspace_members_out.sort(key=lambda member: member["manifest"])
    edges_out.sort(key=_edge_sort_key)
    kind_counts: dict[str, int] = {kind: 0 for kind in sorted(ALLOWED_EDGE_KINDS)}
    target_counts: dict[str, int] = {}
    for edge in edges_out:
        kind_counts[edge["kind"]] += 1
        target_key = "<unconditional>" if edge["target"] is None else edge["target"]
        target_counts[target_key] = target_counts.get(target_key, 0) + 1
    counts = {
        "packages": len(packages_out),
        "registry_packages": sum(item["identity"]["source"] == REGISTRY_SOURCE for item in packages_out),
        "workspace_packages": sum(item["identity"]["source"] == "workspace" for item in packages_out),
        "edges": len(edges_out),
        "edge_kinds": kind_counts,
        "target_specific_edges": sum(edge["target"] is not None for edge in edges_out),
        "target_expressions": len(target_counts) - (1 if "<unconditional>" in target_counts else 0),
    }
    return {
        "workspace": workspace,
        "workspace_members": workspace_members_out,
        "packages": packages_out,
        "edges": edges_out,
        "counts": counts,
    }


def _reject_unsafe_strings(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            _reject_unsafe_strings(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_unsafe_strings(child, f"{path}[{index}]")
    elif isinstance(value, str):
        if "\x00" in value or value.startswith("/") or re.match(r"^[A-Za-z]:[\\/]", value):
            raise ContractError(f"absolute or NUL-containing path/value in policy at {path}: {value!r}")
        if "file://" in value or "/.cargo/" in value or "registry/src" in value:
            raise ContractError(f"Cargo cache/path-dependent value in policy at {path}: {value!r}")


def validate_schema(document: Any, schema: dict[str, Any], label: str = "Rust dependency policy") -> None:
    try:
        from jsonschema import Draft202012Validator, FormatChecker
    except ImportError as exc:
        raise ContractError("jsonschema is required for Rust dependency policy validation") from exc
    try:
        Draft202012Validator.check_schema(schema)
    except Exception as exc:  # jsonschema has several schema error subclasses
        raise ContractError(f"{label} schema is invalid: {exc}") from exc
    errors = sorted(
        Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(document),
        key=lambda error: list(error.path),
    )
    if errors:
        detail = "; ".join(f"{list(error.path)}: {error.message}" for error in errors[:5])
        raise ContractError(f"{label} schema validation failed: {detail}")


def _manifest_package_map(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    packages = manifest.get("packages")
    if not isinstance(packages, list):
        raise ContractError("policy packages must be a list")
    result: dict[str, dict[str, Any]] = {}
    for package in packages:
        if not isinstance(package, dict) or set(package) != {"identity", "checksum", "license", "rust_version", "features"}:
            raise ContractError("policy package record has unknown or missing keys")
        key = package_identity_key(package["identity"])
        if key in result:
            raise ContractError(f"duplicate policy package identity: {key}")
        source = package["identity"]["source"]
        if source not in PACKAGE_SOURCES:
            raise ContractError(f"unknown policy package source: {source!r}")
        if source == REGISTRY_SOURCE:
            if not isinstance(package["checksum"], str) or not re.fullmatch(r"[0-9a-f]{64}", package["checksum"]):
                raise ContractError(f"registry package checksum missing/invalid: {key}")
            if not isinstance(package["license"], str) or not package["license"]:
                raise ContractError(f"registry package license missing: {key}")
        elif package["checksum"] is not None:
            raise ContractError(f"workspace package checksum must be null: {key}")
        if package["rust_version"] is not None and not isinstance(package["rust_version"], str):
            raise ContractError(f"package rust_version is malformed: {key}")
        if not isinstance(package["features"], list) or package["features"] != sorted(set(package["features"])):
            raise ContractError(f"resolved feature set is not a sorted unique list: {key}")
        result[key] = package
    return result


def _validate_policy_semantics(manifest: dict[str, Any]) -> None:
    if manifest.get("schema_version") != EXPECTED_SCHEMA_VERSION or manifest.get("policy_id") != EXPECTED_POLICY_ID:
        raise ContractError("Rust dependency policy identity drifted")
    _reject_unsafe_strings(manifest)
    package_map = _manifest_package_map(manifest)
    if len(package_map) != EXPECTED_PACKAGE_COUNT:
        raise ContractError(f"policy package count drifted: {len(package_map)}")
    if sum(key.startswith("registry:") for key in package_map) != EXPECTED_REGISTRY_PACKAGE_COUNT:
        raise ContractError("policy registry package count drifted")
    if sum(key.startswith("workspace:") for key in package_map) != EXPECTED_WORKSPACE_PACKAGE_COUNT:
        raise ContractError("policy workspace package count drifted")

    members = manifest.get("workspace_members")
    if not isinstance(members, list) or len(members) != EXPECTED_WORKSPACE_PACKAGE_COUNT:
        raise ContractError("policy workspace member count drifted")
    member_keys: set[str] = set()
    for member in members:
        if not isinstance(member, dict) or set(member) != {"identity", "manifest", "targets"}:
            raise ContractError("workspace member record has unknown or missing keys")
        key = package_identity_key(member["identity"])
        if key in member_keys:
            raise ContractError(f"duplicate workspace member identity: {key}")
        member_keys.add(key)
        if not key.startswith("workspace:"):
            raise ContractError(f"workspace member is not a workspace package: {key}")
        path = member["manifest"]
        if not isinstance(path, str) or not path.endswith("/Cargo.toml") or Path(path).is_absolute() or ".." in Path(path).parts:
            raise ContractError(f"workspace member manifest is not a safe relative path: {path!r}")
        if not isinstance(member["targets"], list) or not member["targets"]:
            raise ContractError(f"workspace member has no targets: {key}")
        for target in member["targets"]:
            if not isinstance(target, dict) or set(target) != {"name", "kind", "crate_types", "edition", "required_features", "test", "doc", "doctest"}:
                raise ContractError(f"workspace target record is not closed: {key}")
    if member_keys != {key for key in package_map if key.startswith("workspace:")}:
        raise ContractError("workspace member/package identity sets differ")

    edges = manifest.get("edges")
    if not isinstance(edges, list) or len(edges) != EXPECTED_EDGE_COUNT:
        raise ContractError("policy edge count drifted")
    edge_keys: set[tuple[Any, ...]] = set()
    for edge in edges:
        expected_keys = {"from", "to", "name", "kind", "target", "req", "requested_features", "uses_default_features", "optional", "rename"}
        if not isinstance(edge, dict) or set(edge) != expected_keys:
            raise ContractError("policy edge record has unknown or missing keys")
        if edge["from"] not in package_map or edge["to"] not in package_map:
            raise ContractError(f"policy edge references missing package: {edge!r}")
        if edge["kind"] not in ALLOWED_EDGE_KINDS:
            raise ContractError(f"unknown dependency edge kind: {edge['kind']!r}")
        if edge["target"] is not None and not isinstance(edge["target"], str):
            raise ContractError("dependency edge target expression is malformed")
        if not isinstance(edge["requested_features"], list) or edge["requested_features"] != sorted(set(edge["requested_features"])):
            raise ContractError("dependency edge feature request is not a sorted unique list")
        if not isinstance(edge["uses_default_features"], bool) or not isinstance(edge["optional"], bool):
            raise ContractError("dependency edge booleans are malformed")
        edge_key = (
            edge["from"], edge["to"], edge["name"], edge["kind"], edge["target"], edge["req"],
            tuple(edge["requested_features"]), edge["uses_default_features"], edge["optional"], edge["rename"],
        )
        if edge_key in edge_keys:
            raise ContractError("duplicate policy edge record")
        edge_keys.add(edge_key)

    counts = manifest.get("counts")
    if not isinstance(counts, dict) or set(counts) != {"packages", "registry_packages", "workspace_packages", "edges", "edge_kinds", "target_specific_edges", "target_expressions"}:
        raise ContractError("policy counts are not closed")
    expected_counts = {
        "packages": len(package_map),
        "registry_packages": sum(key.startswith("registry:") for key in package_map),
        "workspace_packages": sum(key.startswith("workspace:") for key in package_map),
        "edges": len(edges),
        "edge_kinds": {kind: sum(edge["kind"] == kind for edge in edges) for kind in sorted(ALLOWED_EDGE_KINDS)},
        "target_specific_edges": sum(edge["target"] is not None for edge in edges),
        "target_expressions": len({edge["target"] for edge in edges if edge["target"] is not None}),
    }
    if counts != expected_counts:
        raise ContractError(f"policy counts are inconsistent: {counts!r} != {expected_counts!r}")

    assertions = manifest.get("feature_assertions")
    if not isinstance(assertions, dict) or set(assertions) != {
        "minijinja", "server_runtime", "tokenizers",
    }:
        raise ContractError("feature assertions are not closed")
    tokenizers = assertions["tokenizers"]
    expected_assertion_keys = {"package", "allowed", "resolved", "forbidden", "required_packages"}
    if not isinstance(tokenizers, dict) or set(tokenizers) != expected_assertion_keys:
        raise ContractError("tokenizers feature assertion is not closed")
    if tokenizers["package"] != TOKENIZERS_PACKAGE:
        raise ContractError("tokenizers feature assertion points at the wrong package")
    if tokenizers["allowed"] != TOKENIZERS_ALLOWED_FEATURES or tokenizers["forbidden"] != TOKENIZERS_FORBIDDEN_FEATURES:
        raise ContractError("tokenizers feature allow/deny policy drifted")
    if tokenizers["resolved"] != TOKENIZERS_ALLOWED_FEATURES:
        raise ContractError("tokenizers resolved features are not exactly [onig]")
    if tokenizers["required_packages"] != [ESAXX_PACKAGE]:
        raise ContractError("tokenizers unconditional esaxx-rs package assertion drifted")
    if package_map[TOKENIZERS_PACKAGE]["features"] != TOKENIZERS_ALLOWED_FEATURES:
        raise ContractError("tokenizers package resolved features are not exactly [onig]")
    if any(feature in package_map[TOKENIZERS_PACKAGE]["features"] for feature in TOKENIZERS_FORBIDDEN_FEATURES):
        raise ContractError("tokenizers has a forbidden resolved feature")
    if ESAXX_PACKAGE not in package_map or package_map[ESAXX_PACKAGE]["features"]:
        raise ContractError("esaxx-rs must remain present with no enabled features")

    minijinja = assertions["minijinja"]
    expected_minijinja_keys = {
        "package", "requested", "resolved", "forbidden", "required_packages",
        "uses_default_features",
    }
    if not isinstance(minijinja, dict) or set(minijinja) != expected_minijinja_keys:
        raise ContractError("MiniJinja feature assertion is not closed")
    if minijinja != {
        "package": MINIJINJA_PACKAGE,
        "requested": MINIJINJA_REQUESTED_FEATURES,
        "resolved": MINIJINJA_RESOLVED_FEATURES,
        "forbidden": MINIJINJA_FORBIDDEN_FEATURES,
        "required_packages": MINIJINJA_REQUIRED_PACKAGES,
        "uses_default_features": False,
    }:
        raise ContractError("MiniJinja feature allow/deny policy drifted")
    if package_map.get(MINIJINJA_PACKAGE, {}).get("features") != MINIJINJA_RESOLVED_FEATURES:
        raise ContractError("MiniJinja resolved features drifted")
    if any(
        feature in package_map[MINIJINJA_PACKAGE]["features"]
        for feature in MINIJINJA_FORBIDDEN_FEATURES
    ):
        raise ContractError("MiniJinja has a forbidden resolved feature")
    for package in MINIJINJA_REQUIRED_PACKAGES:
        if package not in package_map:
            raise ContractError(f"MiniJinja required package is missing: {package}")
    frontend_edges = [
        edge for edge in edges
        if edge["from"] == "workspace:sllm-frontend@0.1.0"
        and edge["to"] == MINIJINJA_PACKAGE
        and edge["kind"] == "normal"
    ]
    if len(frontend_edges) != 1:
        raise ContractError("MiniJinja frontend dependency edge drifted")
    if (
        frontend_edges[0]["requested_features"] != MINIJINJA_REQUESTED_FEATURES
        or frontend_edges[0]["uses_default_features"]
    ):
        raise ContractError("MiniJinja requested feature edge drifted")

    server_runtime = assertions["server_runtime"]
    if server_runtime != {
        "workspace_package": SERVER_PACKAGE,
        "dependencies": SERVER_RUNTIME_DEPENDENCIES,
    }:
        raise ContractError("server runtime feature assertion drifted")
    for dependency in SERVER_RUNTIME_DEPENDENCIES:
        package = dependency["package"]
        if package not in package_map or package_map[package]["features"] != dependency["resolved"]:
            raise ContractError(f"server runtime resolved features drifted: {package}")
        direct_edges = [
            edge for edge in edges
            if edge["from"] == SERVER_PACKAGE and edge["to"] == package and edge["kind"] == "normal"
        ]
        if len(direct_edges) != 1:
            raise ContractError(f"server runtime direct dependency edge drifted: {package}")
        direct = direct_edges[0]
        if direct["requested_features"] != dependency["requested"] or direct["uses_default_features"] != dependency["uses_default_features"]:
            raise ContractError(f"server runtime requested feature edge drifted: {package}")

    msrv = manifest.get("msrv_policy")
    if not isinstance(msrv, dict) or set(msrv) != {"authority_version", "authority_target", "mode", "exceptions"}:
        raise ContractError("MSRV policy is not closed")
    if msrv["authority_version"] != MSRV_AUTHORITY or msrv["authority_target"] != MSRV_TARGET or msrv["mode"] != "current-linux-cargo-check":
        raise ContractError("MSRV authority is not current Linux Rust 1.85")
    exceptions = msrv["exceptions"]
    if not isinstance(exceptions, list) or len(exceptions) != 1:
        raise ContractError("MSRV exception set drifted")
    exception = exceptions[0]
    if set(exception) != {"package", "declared_rust_version", "allowed_target"} or exception != {
        "package": WASIP2_PACKAGE,
        "declared_rust_version": "1.87.0",
        "allowed_target": WASIP2_TARGET,
    }:
        raise ContractError("wasip2 MSRV exception is not the recorded wasm-only exception")
    wasip2 = package_map.get(WASIP2_PACKAGE)
    if wasip2 is None:
        raise ContractError("wasip2 MSRV exception package is missing")
    wasip2_rust_version = wasip2["rust_version"]
    if not isinstance(wasip2_rust_version, str) or wasip2_rust_version != exception["declared_rust_version"]:
        raise ContractError("wasip2 rust_version does not match its MSRV exception")
    for key, package in package_map.items():
        declared = package["rust_version"]
        if declared is None or _rust_version_tuple(declared) <= _rust_version_tuple(MSRV_AUTHORITY):
            continue
        if key != WASIP2_PACKAGE:
            raise ContractError(f"package exceeds Linux MSRV without an explicit exception: {key}")
    incoming = [edge for edge in edges if edge["to"] == WASIP2_PACKAGE]
    if len(incoming) != 1 or incoming[0]["target"] != WASIP2_TARGET:
        raise ContractError("wasip2 must be reachable only through its wasm32-wasip2 target edge")


def validate_manifest_against_observed(
    manifest: dict[str, Any], observed: dict[str, Any], *, schema: dict[str, Any] | None = None
) -> None:
    """Validate policy semantics and exact equality against one observation."""

    if schema is not None:
        validate_schema(manifest, schema)
    _validate_policy_semantics(manifest)
    for section in SECTION_NAMES:
        if manifest.get(section) != observed.get(section):
            raise ContractError(f"Rust dependency {section} graph/field drift detected")


def _cargo_environment() -> dict[str, str]:
    """Return Cargo's offline, host-only B0 environment."""

    environment = os.environ.copy()
    for name in B0_ABSENT_ENVIRONMENT_VARIABLES:
        environment.pop(name, None)
    for name in B0_DISABLED_HIP_FLAGS:
        environment[name] = "0"
    environment["CARGO_NET_OFFLINE"] = "true"
    environment["RUSTUP_AUTO_INSTALL"] = "0"
    return environment


def _cargo_metadata(
    repo: Path,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    command = ["cargo", f"+{MSRV_AUTHORITY}", "metadata", "--locked", "--offline", "--format-version", "1"]
    process = runner(command, cwd=repo, text=True, capture_output=True, check=False, env=_cargo_environment())
    if process.returncode != 0:
        raise ContractError(f"cargo metadata failed ({process.returncode}): {process.stderr.strip()}")
    try:
        import json

        return json.loads(process.stdout)
    except ValueError as exc:
        raise ContractError(f"cargo metadata returned invalid JSON: {exc}") from exc


def run_cargo_check(repo: Path = ROOT, runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run) -> None:
    """Run the exact B0 MSRV check; a command failure is never a pass."""

    command = [
        "cargo", f"+{MSRV_AUTHORITY}", "check", "--jobs", "1", "--workspace", "--all-targets", "--locked", "--offline",
        "--target", MSRV_TARGET,
    ]
    process = runner(command, cwd=repo, text=True, capture_output=True, check=False, env=_cargo_environment())
    if process.returncode != 0:
        detail = (process.stderr or process.stdout or "").strip()
        raise ContractError(f"cargo check failed ({process.returncode}): {detail[-4000:]}")


def validate_policy(repo: Path = ROOT, *, run_check: bool = True) -> None:
    """Validate the checked-in policy against fresh offline Cargo evidence."""

    manifest = read_json(repo / POLICY_PATH)
    schema = read_json(repo / SCHEMA_PATH)
    validate_manifest_against_observed(manifest, {section: manifest.get(section) for section in SECTION_NAMES}, schema=schema)
    metadata = _cargo_metadata(repo)
    observed = normalize_metadata(metadata, repo)
    validate_manifest_against_observed(manifest, observed, schema=schema)
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise ContractError("B0 MSRV authority requires current Linux x86_64 execution")
    if run_check:
        run_cargo_check(repo)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-cargo-check", action="store_true",
        help="skip the final cargo check (for pure local inspection only; the default is fail-closed)",
    )
    args = parser.parse_args(argv)
    try:
        validate_policy(ROOT, run_check=not args.skip_cargo_check)
    except (ContractError, OSError, ValueError) as exc:
        print(f"rust dependency closure: FAIL: {exc}", file=sys.stderr)
        return 1
    print("rust dependency closure: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
