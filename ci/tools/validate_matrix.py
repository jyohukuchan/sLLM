#!/usr/bin/env python3
"""Validate suite, path, pytest-marker, and host-row registration drift."""

from __future__ import annotations

import ast
import fnmatch
import sys
import tomllib
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    ALLOWED_ATTRIBUTES,
    ALLOWED_TIERS,
    DEV_RUST_VERSION,
    MSRV_RUST_VERSION,
    ContractError,
    SAFE_COMMANDS,
    load_manifests,
)

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_HOST_ROWS = {"h0": "tier_h0", "h1": "tier_h1", "h2": "tier_h2"}
H3_SUITE_ID = "h3-compile-only-contract"
H3_STATIC_SUITE_ID = "h0-h3-static-contracts"
G1_STATIC_SUITE_ID = "h0-g0-static-contracts"
EXPECTED_H3_PATH_RULES = {
    "ci/tools/aggregate_h3_results.py",
    "ci/tools/run_h3_compile.py",
    "ci/tools/validate_h3_contracts.py",
    "ci/tools/validate_json_manifests.py",
    "ci/tools/validate_matrix.py",
    "ci/schema/h3-aggregate-v1.schema.json",
    "ci/schema/hip-artifact-metadata-v1.schema.json",
    "ci/schema/rocm-toolchain-v1.schema.json",
    "ci/schema/test-result-v1.schema.json",
    "ci/matrix/hip-compile-v1.json",
    "ci/matrix/suites-v1.json",
    "ci/matrix/path-to-suite-v1.json",
    "ci/toolchains/rocm-7.14.0.json",
    "ci/tests/test_h3_contracts.py",
    "ci/tests/test_h3_runner.py",
    "ci/tests/test_h3_aggregate.py",
    ".github/workflows/h3-compile.yml",
    "crates/sllm-hip-sys/build.rs",
    "native/hip/CMakeLists.txt",
    "native/hip/src/hip_compile_probe.hip.cpp",
}
EXPECTED_FIXTURE_SUITES = {
    "tests/fixtures/api_cases.json": {"h0-python", "h1-host-contract"},
    "tests/fixtures/boundary_cases.json": {"h0-python", "h2-tiny-oracle"},
    "tests/fixtures/kv_layout.json": {"h0-python", "h2-tiny-oracle"},
    "tests/fixtures/sampling_cases.json": {"h0-python", "h2-tiny-oracle"},
}
EXPECTED_H3_STATIC_PATH_RULES = {
    "ci/tests/test_h3_workflow_identity.py",
    ".github/workflows/h3-compile.yml",
}
EXPECTED_G1_STATIC_PATH_RULES = {
    "ci/tools/aggregate_g1_results.py",
    "ci/tools/build_g1_runtime.py",
    "ci/tools/run_g1_evidence.py",
    "ci/tools/validate_g1_contracts.py",
    "ci/tools/validate_json_manifests.py",
    "ci/tools/validate_matrix.py",
    "ci/schema/g1-aggregate-v1.schema.json",
    "ci/schema/g1-report-v1.schema.json",
    "ci/schema/g1-runtime-artifact-v1.schema.json",
    "ci/matrix/g1-runtime-v1.json",
    "ci/matrix/suites-v1.json",
    "ci/matrix/path-to-suite-v1.json",
    "ci/tests/test_g1_builder.py",
    "ci/tests/test_g1_contracts.py",
    "ci/tests/test_g1_runner.py",
}


def selected_suites(path: str, paths: dict[str, object]) -> set[str]:
    result = set(paths["default_suite_ids"])
    for rule in paths["rules"]:
        if fnmatch.fnmatchcase(path, rule["pattern"]):
            result.update(rule["suite_ids"])
    return result


def pytest_markers() -> tuple[set[str], dict[str, set[str]]]:
    pyproject = ROOT / "pyproject.toml"
    if not pyproject.exists():
        raise ContractError("pyproject.toml is missing")
    with pyproject.open("rb") as stream:
        config = tomllib.load(stream)
    marker_lines = config.get("tool", {}).get("pytest", {}).get("ini_options", {}).get("markers", [])
    declared = {str(line).split(":", 1)[0].strip() for line in marker_lines}
    found: dict[str, set[str]] = {}
    for path in sorted((ROOT / "tests").rglob("*.py")):
        relative = path.relative_to(ROOT).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        except (OSError, SyntaxError, UnicodeError) as exc:
            raise ContractError(f"cannot parse test file {relative}: {exc}") from exc
        markers: set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Attribute) or node.attr == "mark":
                continue
            parent = node.value
            if isinstance(parent, ast.Attribute) and parent.attr == "mark" and isinstance(parent.value, ast.Name) and parent.value.id == "pytest":
                markers.add(node.attr)
        if markers:
            found[relative] = markers
    if not {"tier_h1", "tier_h2"}.issubset(declared):
        raise ContractError("required tier_h1/tier_h2 pytest markers are not declared")
    if not any("tier_h1" in markers for markers in found.values()):
        raise ContractError("tier_h1 marker has zero registered tests")
    if not any("tier_h2" in markers for markers in found.values()):
        raise ContractError("tier_h2 marker has zero registered tests")
    return declared, found


def fixture_consumers() -> dict[str, set[str]]:
    """Return every literal ``load_json_fixture`` consumer by fixture path."""

    consumers: dict[str, set[str]] = {path: set() for path in EXPECTED_FIXTURE_SUITES}
    tests_root = ROOT / "tests"
    if not tests_root.is_dir():
        raise ContractError("tests directory is missing")
    known_names = {Path(path).name: path for path in EXPECTED_FIXTURE_SUITES}
    for path in sorted(tests_root.rglob("*.py")):
        relative = path.relative_to(ROOT).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative)
        except (OSError, SyntaxError, UnicodeError) as exc:
            raise ContractError(f"cannot parse fixture consumer {relative}: {exc}") from exc
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            if not isinstance(function, ast.Name) or function.id != "load_json_fixture":
                continue
            if not node.args or not isinstance(node.args[0], ast.Constant) or not isinstance(node.args[0].value, str):
                raise ContractError(f"fixture consumer must name a literal JSON fixture: {relative}")
            name = node.args[0].value
            fixture_path = known_names.get(name)
            if fixture_path is None:
                raise ContractError(f"fixture consumer references an unregistered fixture: {relative}: {name}")
            consumers[fixture_path].add(relative)
    return consumers


def validate_fixture_registration(paths: dict[str, object]) -> None:
    """Require explicit H0 plus H1/H2 ownership for every fixture consumer."""

    rules = {rule["pattern"]: set(rule["suite_ids"]) for rule in paths["rules"]}
    actual_fixtures = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "tests/fixtures").glob("*.json")
        if path.is_file()
    }
    if actual_fixtures != set(EXPECTED_FIXTURE_SUITES):
        raise ContractError(
            "fixture registry drift: "
            f"expected={sorted(EXPECTED_FIXTURE_SUITES)} actual={sorted(actual_fixtures)}"
        )
    for fixture, required_suites in EXPECTED_FIXTURE_SUITES.items():
        if fixture not in rules or rules[fixture] != required_suites:
            raise ContractError(
                f"fixture path must explicitly map to {sorted(required_suites)}: {fixture}"
            )
        if not required_suites.issubset(selected_suites(fixture, paths)):
            raise ContractError(f"fixture path does not select all required suites: {fixture}")

    consumers = fixture_consumers()
    for fixture, required_suites in EXPECTED_FIXTURE_SUITES.items():
        if not consumers[fixture]:
            raise ContractError(f"fixture has no registered consumer: {fixture}")
        for consumer in sorted(consumers[fixture]):
            selected = selected_suites(consumer, paths)
            if not required_suites.issubset(selected):
                raise ContractError(
                    f"fixture consumer is missing its fixture suites: {consumer} -> {fixture}"
                )


def command_is_pytest(argv: list[str], marker: str) -> bool:
    if len(argv) < 5 or argv[0:3] != ["{python}", "-m", "pytest"]:
        return False
    try:
        marker_index = argv.index("-m", 3)
    except ValueError:
        return False
    return marker_index + 1 < len(argv) and argv[marker_index + 1] == marker


def command_is_unittest_script(argv: list[str], script: str) -> bool:
    return argv == ["{python}", script]


def command_is_cargo_workspace_test(argv: list[str]) -> bool:
    return (
        len(argv) >= 4
        and argv[:4] == ["cargo", f"+{DEV_RUST_VERSION}", "test", "--workspace"]
        and "--locked" in argv[4:]
        and "--offline" in argv[4:]
    )


def validate_cargo_toolchain_registration(suites: dict[str, object]) -> None:
    """Allow the development pin everywhere, with one exact MSRV exception."""

    for suite in suites["suites"]:
        for command in suite["commands"]:
            argv = command["argv"]
            if not argv or argv[0] != "cargo":
                continue
            command_id = command["command_id"]
            if command_id == "rust-msrv":
                if argv[1:4] != [f"+{MSRV_RUST_VERSION}", "check", "--workspace"]:
                    raise ContractError("rust-msrv must be exactly cargo +1.85.0 check --workspace")
            elif len(argv) < 2 or argv[1] != f"+{DEV_RUST_VERSION}":
                raise ContractError(
                    f"normal Rust command must select +{DEV_RUST_VERSION}: {suite['suite_id']}.{command_id}"
                )


def main() -> int:
    try:
        from validate_g1_contracts import validate_g1_matrix

        validate_g1_matrix(ROOT)
        suites, host, paths = load_manifests(ROOT)
        suite_by_id = {suite["suite_id"]: suite for suite in suites["suites"]}
        if set(suites) != {"schema_version", "registry_id", "revision", "allowed_tiers", "allowed_attributes", "suites"}:
            raise ContractError("suites-v1 has unknown or missing top-level key")
        if set(host) != {"schema_version", "matrix_id", "revision", "rows"}:
            raise ContractError("host-v1 has unknown or missing top-level key")
        if set(paths) != {"schema_version", "revision", "default_suite_ids", "rules"}:
            raise ContractError("path-to-suite-v1 has unknown or missing top-level key")
        for suite in suites["suites"]:
            sid = suite["suite_id"]
            if set(suite) != {"suite_id", "tier", "marker", "attributes", "test_ids", "commands"}:
                raise ContractError(f"suite {sid} has unknown or missing key")
            if set(suite["attributes"]) != set(ALLOWED_ATTRIBUTES):
                raise ContractError(f"suite {sid} has unknown attribute")
            if suite["marker"] not in ALLOWED_TIERS or suite["marker"] != suite["tier"]:
                raise ContractError(f"invalid marker/tier for {sid}")
            if not suite["test_ids"] or len(set(suite["test_ids"])) != len(suite["test_ids"]):
                raise ContractError(f"zero or duplicate test registration in {sid}")
            if not suite["commands"]:
                raise ContractError(f"zero command collection in {sid}")
            command_ids: set[str] = set()
            for command in suite["commands"]:
                if set(command) != {"command_id", "argv"} or command["command_id"] in command_ids:
                    raise ContractError(f"invalid or duplicate command in {sid}")
                command_ids.add(command["command_id"])
                argv = command["argv"]
                if not isinstance(argv, list) or not argv or not all(isinstance(arg, str) and arg for arg in argv):
                    raise ContractError(f"invalid command argv in {sid}")
                executable = argv[0]
                if "{python}" not in argv and executable not in SAFE_COMMANDS:
                    raise ContractError(f"non-allowlisted command in {sid}: {executable}")
                if any(token in " ".join(argv).lower() for token in ("gpu", "model", "network", "fallback")):
                    raise ContractError(f"host registry command mentions prohibited host capability in {sid}")
            if suite["tier"] in {"tier_h0", "tier_h1", "tier_h2"} and any(suite["attributes"][key] for key in ("requires_gpu", "requires_model", "network", "quarantined")):
                raise ContractError(f"required host suite has prohibited attribute: {sid}")

        h3_suite = suite_by_id.get(H3_SUITE_ID)
        if h3_suite is None:
            raise ContractError(f"missing independent H3 suite: {H3_SUITE_ID}")
        if h3_suite["tier"] != "tier_h3" or h3_suite["marker"] != "tier_h3":
            raise ContractError("H3 compile suite has the wrong tier/marker")
        if h3_suite["attributes"] != {key: False for key in ALLOWED_ATTRIBUTES}:
            raise ContractError("H3 compile suite must be model-free, GPU-free, offline, and non-quarantined")
        if h3_suite["commands"] != [{"command_id": "h3-compile-contract", "argv": ["{python}", "-m", "unittest", "ci.tests.test_h3_contracts", "ci.tests.test_h3_runner", "ci.tests.test_h3_aggregate"]}]:
            raise ContractError("H3 compile suite command registration drifted")
        h3_static_suite = suite_by_id.get(H3_STATIC_SUITE_ID)
        if h3_static_suite is None:
            raise ContractError(f"missing required H3 static suite: {H3_STATIC_SUITE_ID}")
        if h3_static_suite["tier"] != "tier_h0" or h3_static_suite["marker"] != "tier_h0":
            raise ContractError("H3 static suite has the wrong tier/marker")
        if h3_static_suite["commands"] != [
            {"command_id": "h3-static-contracts", "argv": ["{python}", "ci/tests/test_h3_contracts.py"]},
            {"command_id": "h3-workflow-identity", "argv": ["{python}", "ci/tests/test_h3_workflow_identity.py"]},
        ]:
            raise ContractError("H3 workflow identity test is not registered in the required H0 suite")

        g1_static_suite = suite_by_id.get(G1_STATIC_SUITE_ID)
        if g1_static_suite is None or g1_static_suite["tier"] != "tier_h0" or g1_static_suite["marker"] != "tier_h0":
            raise ContractError("G1 static contract test must be registered in H0")
        expected_g1_commands = [
            {"command_id": "g1-static-contracts", "argv": ["{python}", "ci/tests/test_g1_contracts.py"]},
            {"command_id": "g1-builder", "argv": ["{python}", "ci/tests/test_g1_builder.py"]},
            {"command_id": "g1-runner", "argv": ["{python}", "ci/tests/test_g1_runner.py"]},
        ]
        if not all(command in g1_static_suite["commands"] for command in expected_g1_commands):
            raise ContractError("G1 static suite must collect contracts, builder, and runner tests")

        validate_cargo_toolchain_registration(suites)

        rows = {row["row_id"]: row for row in host["rows"]}
        if rows.keys() != EXPECTED_HOST_ROWS.keys():
            raise ContractError("host rows must be exactly h0, h1, h2")
        for row_id, row in rows.items():
            if row["tier"] != EXPECTED_HOST_ROWS[row_id] or row["required"] is not True or not row["suite_ids"]:
                raise ContractError(f"invalid required host row: {row_id}")
            for sid in row["suite_ids"]:
                if sid not in suite_by_id or suite_by_id[sid]["tier"] != row["tier"]:
                    raise ContractError(f"row {row_id} references unknown/wrong-tier suite {sid}")
                if sid == H3_SUITE_ID:
                    raise ContractError("H3 suite must not be registered in host-required rows")
        h1 = suite_by_id["h1-host-contract"]
        h2 = suite_by_id["h2-tiny-oracle"]
        h0_self_test = suite_by_id["h0-self-test"]
        if not any(command_is_cargo_workspace_test(command["argv"]) for command in h1["commands"]):
            raise ContractError("h1 must collect cargo workspace tests")
        if not any(command_is_pytest(command["argv"], "tier_h1") for command in h1["commands"]):
            raise ContractError("h1 must collect pytest marker tier_h1")
        if not any(command_is_unittest_script(command["argv"], "ci/tests/test_h1_contracts.py") for command in h1["commands"]):
            raise ContractError("h1 must collect the registered CI contract unittest module")
        if not any(command_is_unittest_script(command["argv"], "ci/tests/test_fail_closed.py") for command in h0_self_test["commands"]):
            raise ContractError("h0 must collect the registered fail-closed unittest module")
        if len(h2["commands"]) != 2 or not any(command_is_pytest(command["argv"], "tier_h2") for command in h2["commands"]):
            raise ContractError("h2 must collect pytest marker tier_h2")
        if not any(command_is_unittest_script(command["argv"], "ci/tests/test_h2_oracle.py") for command in h2["commands"]):
            raise ContractError("h2 must collect the registered CI oracle unittest module")

        if len(paths["default_suite_ids"]) != len(set(paths["default_suite_ids"])) or not paths["default_suite_ids"]:
            raise ContractError("path mapping has zero or duplicate default suites")
        patterns: set[str] = set()
        for rule in paths["rules"]:
            if set(rule) != {"pattern", "suite_ids"} or rule["pattern"] in patterns:
                raise ContractError("path rule has unknown keys, zero suites, or duplicate pattern")
            patterns.add(rule["pattern"])
            if not rule["suite_ids"] or len(rule["suite_ids"]) != len(set(rule["suite_ids"])):
                raise ContractError(f"path rule has zero/duplicate suites: {rule['pattern']}")
            if any(sid not in suite_by_id for sid in rule["suite_ids"]):
                raise ContractError(f"path rule references unknown suite: {rule['pattern']}")
        rules_by_pattern = {rule["pattern"]: set(rule["suite_ids"]) for rule in paths["rules"]}
        for h3_path in EXPECTED_H3_PATH_RULES:
            if H3_SUITE_ID not in rules_by_pattern.get(h3_path, set()):
                raise ContractError(f"H3 path is not explicitly registered to {H3_SUITE_ID}: {h3_path}")
        for h3_static_path in EXPECTED_H3_STATIC_PATH_RULES:
            if H3_STATIC_SUITE_ID not in rules_by_pattern.get(h3_static_path, set()):
                raise ContractError(
                    f"H3 workflow identity path is not explicitly registered to {H3_STATIC_SUITE_ID}: {h3_static_path}"
                )
        for g1_path in EXPECTED_G1_STATIC_PATH_RULES:
            if G1_STATIC_SUITE_ID not in rules_by_pattern.get(g1_path, set()):
                raise ContractError(f"G1 path is not explicitly registered to the H0 static suite: {g1_path}")

        declared_markers, marked_files = pytest_markers()
        known_suite_markers = {suite["marker"] for suite in suites["suites"]}
        if not set().union(*(markers for markers in marked_files.values())) <= declared_markers:
            raise ContractError("test uses an undeclared pytest marker")
        for relative, markers in marked_files.items():
            selected = selected_suites(relative, paths)
            if "tier_h1" in markers and "h1-host-contract" not in selected:
                raise ContractError(f"tier_h1 test path is not mapped to h1: {relative}")
            if "tier_h2" in markers and "h2-tiny-oracle" not in selected:
                raise ContractError(f"tier_h2 test path is not mapped to h2: {relative}")
            if markers - known_suite_markers - {"requires_gpu", "requires_model", "slow", "network", "quarantined"}:
                raise ContractError(f"test marker is not registered in suites: {relative}")

        validate_fixture_registration(paths)

        actual_paths: list[str] = []
        for root_name in ("ci", "crates", "native", "include", "tests", ".github"):
            root = ROOT / root_name
            if root.exists():
                actual_paths.extend(path.relative_to(ROOT).as_posix() for path in root.rglob("*") if path.is_file() and "__pycache__" not in path.parts)
        for root_name in ("Cargo.toml", "Cargo.lock", "pyproject.toml"):
            if (ROOT / root_name).exists():
                actual_paths.append(root_name)
        for relative in sorted(actual_paths):
            selected = selected_suites(relative, paths)
            if not selected:
                raise ContractError(f"path maps to zero suites: {relative}")
            if relative == "Cargo.toml" or relative.startswith("crates/") or relative.endswith(".rs"):
                if not {"h0-rust-format", "h0-rust-clippy"}.issubset(selected):
                    raise ContractError(f"Rust path is missing H0 Rust suites: {relative}")
            if relative.startswith(("native/", "include/")) or relative.endswith((".cpp", ".hpp", ".h")):
                if "h0-cpp-format-static" not in selected:
                    raise ContractError(f"native path is missing H0 C++ suite: {relative}")
            if relative.startswith(".github/") and "h0-json-schema-manifest-workflow" not in selected:
                raise ContractError(f"workflow path is not mapped to schema/workflow suite: {relative}")

        print(f"matrix registration: PASS suites={len(suites['suites'])} rows={len(rows)} marked_files={len(marked_files)} attrs={','.join(ALLOWED_ATTRIBUTES)}")
        return 0
    except (ContractError, OSError, ValueError, tomllib.TOMLDecodeError) as exc:
        print(f"matrix registration: FAIL: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
