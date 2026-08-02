"""Shared, deterministic helpers for the Phase 1 host contract tools."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = ROOT / "ci" / "schema"
MATRIX_DIR = ROOT / "ci" / "matrix"

ALLOWED_TIERS = (
    "tier_h0", "tier_h1", "tier_h2", "tier_h3", "tier_g0", "tier_g1",
    "tier_g2", "tier_g3", "tier_g4", "tier_p0", "tier_p1",
)
ALLOWED_ATTRIBUTES = (
    "requires_gpu", "requires_model", "slow", "network", "quarantined",
)
STATES = ("PASS", "FAIL", "SKIP", "INFRA_ERROR", "QUARANTINED")
SAFE_COMMANDS = {"cargo", "cmake", "clang-format"}
COUNT_KEYS = ("collected", "selected", "passed", "failed", "skipped", "deselected")
HOST_PYTHON_VERSION = "3.12.10"
MSRV_RUST_VERSION = "1.85.0"
DEV_RUST_VERSION = "1.97.1"
SHA40 = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
LOCKED_REQUIREMENT = re.compile(
    r"^([A-Za-z0-9][A-Za-z0-9_.-]*)==([^\s;\\]+)\s*\\?$"
)
LOCKED_HASH = re.compile(r"^--hash=sha256:([0-9a-f]{64})\s*\\?$")


class ContractError(ValueError):
    """A malformed local contract or result; callers should fail closed."""


def validate_cargo_command(
    argv: list[str],
    *,
    label: str,
    allow_msrv: bool = False,
) -> None:
    """Require an explicit Rust pin and confine MSRV to ``cargo check``."""

    if len(argv) < 3:
        raise ContractError(f"Cargo command is incomplete for {label}")
    selector, subcommand = argv[1:3]
    if selector == f"+{DEV_RUST_VERSION}":
        return
    if (
        allow_msrv
        and selector == f"+{MSRV_RUST_VERSION}"
        and subcommand == "check"
    ):
        return
    expected = f"+{DEV_RUST_VERSION}"
    if allow_msrv:
        expected += f" (or +{MSRV_RUST_VERSION} for the MSRV check only)"
    raise ContractError(f"Cargo command does not select {expected} for {label}")


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_bytes(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, ValueError) as exc:
        raise ContractError(f"cannot read JSON {path}: {exc}") from exc


def load_hygiene_allowlist(repo: Path = ROOT) -> list[dict[str, Any]]:
    """Load the closed, reviewed allowlist used by both hygiene checks."""
    document = read_json(repo / "ci/policy/hygiene-allowlist-v1.json")
    schema = read_json(repo / "ci/schema/hygiene-allowlist-v1.schema.json")
    try:
        from jsonschema import Draft202012Validator, FormatChecker
    except ImportError as exc:
        raise ContractError("jsonschema is required for hygiene allowlist validation") from exc
    errors = sorted(Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(document), key=lambda error: list(error.path))
    if errors:
        raise ContractError("hygiene allowlist schema invalid: " + "; ".join(error.message for error in errors[:5]))
    entries = document["entries"]
    seen: set[str] = set()
    for entry in entries:
        path = entry["path"]
        if path in seen or "\x00" in path or path.startswith("/") or ".." in Path(path).parts:
            raise ContractError(f"duplicate or unsafe hygiene allowlist path: {path}")
        seen.add(path)
        if entry.get("max_count", 1) < 1:
            raise ContractError(f"invalid hygiene allowlist max_count: {path}")
    return entries


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def iso_z(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def parse_time(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ContractError(f"invalid RFC3339 timestamp: {value}") from exc
    if parsed.tzinfo is None:
        raise ContractError(f"timestamp must include timezone: {value}")
    return parsed.astimezone(timezone.utc)


def run_git(args: list[str], repo: Path = ROOT, *, check: bool = True) -> str:
    proc = subprocess.run(["git", *args], cwd=repo, text=True, capture_output=True, check=False)
    if check and proc.returncode != 0:
        raise ContractError(f"git {' '.join(args)} failed: {proc.stderr.strip()}")
    return proc.stdout.strip()


def identity(repo: Path = ROOT) -> dict[str, str]:
    commit = run_git(["rev-parse", "HEAD"], repo)
    tree = run_git(["rev-parse", "HEAD^{tree}"], repo)
    if len(commit) != 40 or any(c not in "0123456789abcdef" for c in commit):
        raise ContractError(f"not an exact commit SHA: {commit}")
    if len(tree) != 40 or any(c not in "0123456789abcdef" for c in tree):
        raise ContractError(f"not an exact tree OID: {tree}")
    return {"commit": commit, "tree": tree}


def exact_sha(value: str | None, name: str) -> str:
    """Require a complete, lowercase Git object identity."""
    if not isinstance(value, str) or not SHA40.fullmatch(value):
        raise ContractError(f"{name} must be a 40-character lowercase SHA")
    return value


def worktree_status(repo: Path = ROOT) -> dict[str, list[str]]:
    """Return every non-ignored mutation without ever modifying the checkout."""
    tracked = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=no"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    if tracked.returncode != 0 or untracked.returncode != 0:
        detail = (tracked.stderr or untracked.stderr).strip()
        raise ContractError(f"cannot inspect worktree status: {detail}")
    return {
        "tracked": [line for line in tracked.stdout.splitlines() if line],
        "untracked": [line for line in untracked.stdout.splitlines() if line],
    }


def ensure_clean_worktree(repo: Path = ROOT) -> None:
    status = worktree_status(repo)
    dirty = status["tracked"] + status["untracked"]
    if dirty:
        preview = ", ".join(dirty[:8])
        suffix = "" if len(dirty) <= 8 else ", ..."
        raise ContractError(f"required CI checkout is not clean: {preview}{suffix}")


def fixture_size_bytes(repo: Path = ROOT) -> int:
    """Measure tiny fixture inputs while excluding interpreter cache output."""
    fixture_root = repo / "tests" / "fixtures"
    if not fixture_root.is_dir():
        raise ContractError("tests/fixtures is missing")
    resolved_root = fixture_root.resolve()
    total = 0
    for path in fixture_root.rglob("*"):
        if "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        if path.is_file():
            if path.is_symlink() or not path.resolve().is_relative_to(resolved_root):
                raise ContractError(f"fixture escapes fixture root: {path}")
            try:
                total += path.stat().st_size
            except OSError as exc:
                raise ContractError(f"cannot stat fixture {path}: {exc}") from exc
    return total


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _canonical_package_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def locked_host_packages(repo: Path = ROOT) -> dict[str, str]:
    """Return the exact, hashed host lock snapshot after validating direct deps.

    The lock contains both direct and transitive packages.  Direct project test
    dependencies must use an exact ``==`` pin and occur in that larger snapshot
    at the same version; transitive entries are intentionally retained.
    """

    lock_path = repo / "ci/requirements-host.txt"
    try:
        lines = lock_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ContractError(f"cannot read host dependency lock {lock_path}: {exc}") from exc

    locked: dict[str, str] = {}
    display_names: dict[str, str] = {}
    hashes: dict[str, set[str]] = {}
    current: str | None = None
    for line_number, raw in enumerate(lines, start=1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("--") and not stripped.startswith("--hash="):
            if stripped != "--only-binary=:all:":
                raise ContractError(
                    f"unsupported host lock option at {lock_path}:{line_number}: {stripped}"
                )
            current = None
            continue
        requirement = LOCKED_REQUIREMENT.fullmatch(stripped)
        if requirement:
            display_name, version = requirement.groups()
            canonical = _canonical_package_name(display_name)
            if canonical in locked:
                raise ContractError(f"duplicate host lock package: {display_name}")
            locked[canonical] = version
            display_names[canonical] = display_name
            hashes[canonical] = set()
            current = canonical
            continue
        digest = LOCKED_HASH.fullmatch(stripped)
        if digest and current is not None:
            hashes[current].add(digest.group(1))
            continue
        raise ContractError(
            f"host lock is not an exact hashed snapshot at "
            f"{lock_path}:{line_number}: {stripped}"
        )
    if not locked:
        raise ContractError("host dependency lock contains zero packages")
    unhashed = sorted(display_names[name] for name, values in hashes.items() if not values)
    if unhashed:
        raise ContractError(
            "host dependency lock entries have no SHA-256 hash: " + ", ".join(unhashed)
        )

    pyproject_path = repo / "pyproject.toml"
    try:
        with pyproject_path.open("rb") as stream:
            project = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ContractError(f"cannot read {pyproject_path}: {exc}") from exc
    project_table = project.get("project")
    if not isinstance(project_table, dict):
        raise ContractError("pyproject.toml has no [project] table")
    direct_values: list[Any] = list(project_table.get("dependencies", []))
    optional = project_table.get("optional-dependencies", {})
    if not isinstance(optional, dict):
        raise ContractError("pyproject optional-dependencies must be a table")
    direct_values.extend(optional.get("test", []))
    for value in direct_values:
        if not isinstance(value, str):
            raise ContractError("direct host dependency must be a string")
        match = re.fullmatch(
            r"([A-Za-z0-9][A-Za-z0-9_.-]*)(?:\[[^]]+\])?==([^\s;]+)",
            value,
        )
        if not match:
            raise ContractError(
                f"direct host dependency is not an exact unconditional pin: {value}"
            )
        name, version = match.groups()
        canonical = _canonical_package_name(name)
        if locked.get(canonical) != version:
            raise ContractError(
                f"direct host dependency {name}=={version} does not match exact lock snapshot"
            )
    return {name: locked[name] for name in sorted(locked)}


def toolchain_snapshot(repo: Path = ROOT) -> dict[str, Any]:
    def version(executable: str, args: list[str]) -> str:
        try:
            found = subprocess.run(
                [executable, *args],
                cwd=repo,
                text=True,
                capture_output=True,
                check=False,
                env={**os.environ, "RUSTUP_AUTO_INSTALL": "0"},
            )
        except OSError:
            return "unavailable"
        if found.returncode != 0:
            return "unavailable"
        line = (found.stdout or found.stderr).splitlines()
        return line[0].strip() if line else "unavailable"

    locked_packages = locked_host_packages(repo)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(aliased=True),
        "system": platform.system(),
        "machine": platform.machine(),
        "git": version("git", ["--version"]),
        "rustc_dev": version(
            "rustup", ["run", DEV_RUST_VERSION, "rustc", "--version"]
        ),
        "cargo_dev": version(
            "rustup", ["run", DEV_RUST_VERSION, "cargo", "--version"]
        ),
        "rustc_msrv": version(
            "rustup", ["run", MSRV_RUST_VERSION, "rustc", "--version"]
        ),
        "cargo_msrv": version(
            "rustup", ["run", MSRV_RUST_VERSION, "cargo", "--version"]
        ),
        "clang_format": version("clang-format", ["--version"]),
        "cmake": version("cmake", ["--version"]),
        "host_packages": {
            package: package_version(package) for package in locked_packages
        },
    }


def validate_required_toolchain(
    snapshot: dict[str, Any],
    *,
    require_dev_rust: bool,
    require_msrv_rust: bool,
    repo: Path = ROOT,
) -> None:
    """Reject an unpinned CI toolchain instead of merely recording it."""
    expected_prefixes: dict[str, str] = {}
    if require_dev_rust:
        expected_prefixes.update(
            {
                "rustc_dev": f"rustc {DEV_RUST_VERSION} ",
                "cargo_dev": f"cargo {DEV_RUST_VERSION} ",
            }
        )
    if require_msrv_rust:
        expected_prefixes.update(
            {
                "rustc_msrv": f"rustc {MSRV_RUST_VERSION} ",
                "cargo_msrv": f"cargo {MSRV_RUST_VERSION} ",
            }
        )
    if snapshot.get("python") != HOST_PYTHON_VERSION:
        raise ContractError(
            f"required Python is {HOST_PYTHON_VERSION}, got {snapshot.get('python')!r}"
        )
    if snapshot.get("system") != "Linux" or snapshot.get("machine") != "x86_64":
        raise ContractError(
            "required host platform is Linux x86_64, got "
            f"{snapshot.get('system')!r}/{snapshot.get('machine')!r}"
        )
    for field, prefix in expected_prefixes.items():
        if not snapshot.get(field, "").startswith(prefix):
            raise ContractError(f"required {field} is {prefix.strip()}, got {snapshot.get(field)!r}")
    expected_packages = locked_host_packages(repo)
    packages = snapshot.get("host_packages")
    if not isinstance(packages, dict) or set(packages) != set(expected_packages):
        raise ContractError("required host package snapshot is incomplete")
    for package, expected in expected_packages.items():
        if packages.get(package) != expected:
            raise ContractError(
                f"required {package} is {expected}, got {packages.get(package)!r}"
            )


def load_manifests(repo: Path = ROOT) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    suites = read_json(repo / "ci/matrix/suites-v1.json")
    host = read_json(repo / "ci/matrix/host-v1.json")
    paths = read_json(repo / "ci/matrix/path-to-suite-v1.json")
    if set(suites) != {"schema_version", "registry_id", "revision", "allowed_tiers", "allowed_attributes", "suites"}:
        raise ContractError("suites-v1 has unknown or missing top-level key")
    if set(host) != {"schema_version", "matrix_id", "revision", "rows"}:
        raise ContractError("host-v1 has unknown or missing top-level key")
    if set(paths) != {"schema_version", "revision", "default_suite_ids", "rules"}:
        raise ContractError("path-to-suite-v1 has unknown or missing top-level key")
    if suites.get("schema_version") != "suites-v1":
        raise ContractError("wrong suites manifest version")
    if host.get("schema_version") != "host-v1":
        raise ContractError("wrong host manifest version")
    if paths.get("schema_version") != "path-to-suite-v1":
        raise ContractError("wrong path-to-suite manifest version")
    if host.get("matrix_id") != "host-required":
        raise ContractError("host matrix must be host-required")
    rows = host.get("rows")
    if not isinstance(rows, list) or [row.get("row_id") for row in rows] != ["h0", "h1", "h2"]:
        raise ContractError("host-v1 must contain exactly rows h0, h1, h2 in order")
    if len({row.get("row_id") for row in rows}) != 3:
        raise ContractError("duplicate host row")
    if suites.get("allowed_tiers") != list(ALLOWED_TIERS):
        raise ContractError("suite tier allowlist drift")
    if suites.get("allowed_attributes") != list(ALLOWED_ATTRIBUTES):
        raise ContractError("suite attribute allowlist drift")
    suite_items = suites.get("suites")
    if not isinstance(suite_items, list) or not suite_items:
        raise ContractError("suite registry has zero suites")
    by_id: dict[str, dict[str, Any]] = {}
    seen_tests: set[str] = set()
    for suite in suite_items:
        sid = suite.get("suite_id")
        if not isinstance(sid, str) or sid in by_id:
            raise ContractError(f"missing or duplicate suite_id: {sid!r}")
        if set(suite) != {"suite_id", "tier", "marker", "attributes", "test_ids", "commands"}:
            raise ContractError(f"suite {sid!r} has unknown or missing key")
        if suite.get("tier") not in ALLOWED_TIERS or suite.get("marker") != suite.get("tier"):
            raise ContractError(f"invalid suite tier/marker: {sid}")
        attrs = suite.get("attributes")
        if not isinstance(attrs, dict) or set(attrs) != set(ALLOWED_ATTRIBUTES) or any(not isinstance(attrs[key], bool) for key in ALLOWED_ATTRIBUTES):
            raise ContractError(f"invalid attributes for {sid}")
        if attrs["requires_gpu"] != suite["tier"].startswith(("tier_g", "tier_p")):
            raise ContractError(f"GPU attribute mismatch for {sid}")
        tests = suite.get("test_ids")
        commands = suite.get("commands")
        if not isinstance(tests, list) or not tests or any(not isinstance(item, str) for item in tests):
            raise ContractError(f"zero or invalid test registration for {sid}")
        for test_id in tests:
            if test_id in seen_tests:
                raise ContractError(f"duplicate test registration: {test_id}")
            seen_tests.add(test_id)
        if not isinstance(commands, list) or not commands:
            raise ContractError(f"zero command collection for {sid}")
        for command in commands:
            if set(command) != {"command_id", "argv"}:
                raise ContractError(f"invalid command keys for {sid}")
            if not isinstance(command.get("command_id"), str) or not isinstance(command.get("argv"), list) or not command["argv"] or any(not isinstance(arg, str) or not arg for arg in command["argv"]):
                raise ContractError(f"invalid command registration for {sid}")
            executable = command["argv"][0]
            if executable != "{python}" and executable not in SAFE_COMMANDS:
                raise ContractError(f"command executable is not allowlisted for {sid}: {executable}")
            if executable == "cargo":
                validate_cargo_command(
                    command["argv"],
                    label=sid,
                    allow_msrv=command["command_id"] == "rust-msrv",
                )
        by_id[sid] = suite
    for row in rows:
        if set(row) != {
            "row_id", "tier", "required", "suite_ids", "seed", "timeout_seconds",
            "max_command_seconds", "max_rss_bytes", "fixture_size_limit_bytes",
            "max_command_output_bytes", "max_row_output_bytes", "address_space_limit_bytes",
        }:
            raise ContractError(f"host row has unknown or missing key: {row.get('row_id')!r}")
        if row.get("tier") not in ALLOWED_TIERS or not row.get("required"):
            raise ContractError(f"invalid host row: {row}")
        if (
            not isinstance(row.get("seed"), int)
            or isinstance(row.get("seed"), bool)
            or not 0 <= row["seed"] <= 0xFFFFFFFF
        ):
            raise ContractError(f"invalid seed for {row['row_id']}")
        for field in (
            "timeout_seconds", "max_command_seconds", "max_rss_bytes",
            "fixture_size_limit_bytes", "max_command_output_bytes", "max_row_output_bytes",
        ):
            if not isinstance(row.get(field), int) or row[field] <= 0:
                raise ContractError(f"invalid {field} for {row['row_id']}")
        if row["max_command_seconds"] > row["timeout_seconds"]:
            raise ContractError(f"command timeout exceeds row timeout for {row['row_id']}")
        if row["max_row_output_bytes"] < row["max_command_output_bytes"]:
            raise ContractError(f"row output cap is smaller than command cap for {row['row_id']}")
        address_space_limit = row.get("address_space_limit_bytes")
        if address_space_limit is not None and (
            not isinstance(address_space_limit, int) or address_space_limit <= 0
        ):
            raise ContractError(f"invalid address_space_limit_bytes for {row['row_id']}")
        if row["row_id"] == "h2" and address_space_limit != 4 * 1024 * 1024 * 1024:
            raise ContractError("h2 must retain its 4 GiB address-space limit")
        if row["row_id"] != "h2" and address_space_limit is not None:
            raise ContractError(f"only h2 may define an address-space limit: {row['row_id']}")
        if not isinstance(row.get("suite_ids"), list) or not row["suite_ids"]:
            raise ContractError(f"zero suite collection for {row['row_id']}")
        for sid in row["suite_ids"]:
            if sid not in by_id or by_id[sid]["tier"] != row["tier"]:
                raise ContractError(f"row {row['row_id']} references unknown/wrong-tier suite {sid}")
    if set(paths.get("default_suite_ids", [])) - set(by_id):
        raise ContractError("path mapping references unknown default suite")
    if (
        not isinstance(paths.get("default_suite_ids"), list)
        or not paths["default_suite_ids"]
        or len(paths["default_suite_ids"]) != len(set(paths["default_suite_ids"]))
    ):
        raise ContractError("path mapping has zero default suites")
    rules = paths.get("rules")
    if not isinstance(rules, list) or not rules:
        raise ContractError("path mapping has zero rules")
    patterns: set[str] = set()
    for rule in rules:
        if not isinstance(rule, dict) or set(rule) != {"pattern", "suite_ids"}:
            raise ContractError("path mapping rule has unknown or missing key")
        if not isinstance(rule.get("pattern"), str) or not rule.get("pattern"):
            raise ContractError("empty path mapping pattern")
        if rule["pattern"] in patterns:
            raise ContractError(f"duplicate path mapping pattern: {rule['pattern']}")
        patterns.add(rule["pattern"])
        if (
            not isinstance(rule.get("suite_ids"), list)
            or not rule["suite_ids"]
            or len(rule["suite_ids"]) != len(set(rule["suite_ids"]))
            or set(rule["suite_ids"]) - set(by_id)
        ):
            raise ContractError("path mapping references unknown or zero suites")
    return suites, host, paths


def matrix_manifest_hash(repo: Path = ROOT) -> str:
    _, host, _ = load_manifests(repo)
    return sha256_json(host)


def manifest_bundle_hash(repo: Path = ROOT) -> str:
    suites, host, paths = load_manifests(repo)
    return sha256_json({"host-v1": host, "path-to-suite-v1": paths, "suites-v1": suites})


def tuple_digest(row: dict[str, Any]) -> str:
    return sha256_json(row)


def validate_result_payload(payload: dict[str, Any], schema_path: Path | None = None) -> None:
    schema_path = schema_path or SCHEMA_DIR / "test-result-v1.schema.json"
    try:
        from jsonschema import Draft202012Validator, FormatChecker
    except ImportError as exc:
        raise ContractError("jsonschema is required for result validation") from exc
    schema = read_json(schema_path)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(payload), key=lambda error: list(error.path))
    if errors:
        detail = "; ".join(f"{'.'.join(map(str, error.path)) or '<root>'}: {error.message}" for error in errors[:8])
        raise ContractError(f"result schema invalid: {detail}")
    counts = payload["counts"]
    cases = payload["cases"]
    steps = payload["steps"]
    resource_summary = payload["resource"]
    if payload["command_sha256"] != sha256_json(payload["command"]):
        raise ContractError("command hash does not match the command manifest")
    if payload["toolchain_sha256"] != sha256_json(payload["toolchain"]):
        raise ContractError("toolchain hash does not match the toolchain snapshot")
    if payload["artifact"]["content_sha256"] != command_content_hash(steps):
        raise ContractError("artifact content hash does not match the step records")

    def validate_counts(value: dict[str, int], label: str) -> None:
        if value["collected"] != value["selected"] + value["deselected"]:
            raise ContractError(
                f"{label}: collected must equal selected plus deselected"
            )
        if value["selected"] != (
            value["passed"] + value["failed"] + value["skipped"]
        ):
            raise ContractError(
                f"{label}: selected count does not match executed outcomes"
            )

    def validate_step_resource(
        value: dict[str, Any], label: str, duration: float, state: str
    ) -> None:
        if value["output_bytes"] != value["stdout_bytes"] + value["stderr_bytes"]:
            raise ContractError(f"{label}: output byte total does not match streams")
        if value["captured_output_bytes"] != (
            value["stdout_captured_bytes"] + value["stderr_captured_bytes"]
        ):
            raise ContractError(
                f"{label}: captured output byte total does not match streams"
            )
        if (
            value["stdout_captured_bytes"] > value["stdout_bytes"]
            or value["stderr_captured_bytes"] > value["stderr_bytes"]
            or value["captured_output_bytes"] > value["output_limit_bytes"]
        ):
            raise ContractError(f"{label}: captured output is not bounded")
        expected_output_breach = value["output_bytes"] > value["output_limit_bytes"]
        if value["output_breach"] != expected_output_breach:
            raise ContractError(f"{label}: inconsistent command output breach flag")
        expected_rss_breach = value["max_rss_bytes"] > value["max_rss_limit_bytes"]
        if value["rss_breach"] != expected_rss_breach:
            raise ContractError(f"{label}: inconsistent max RSS breach flag")
        if value["address_space_limit_bytes"] is None:
            if value["address_space_limit_enforced"]:
                raise ContractError(f"{label}: null address-space limit was reported enforced")
        elif state == "PASS" and not value["address_space_limit_enforced"]:
            raise ContractError(f"{label}: address-space limit was not enforced")
        if state == "PASS" and (
            value["output_breach"]
            or value["rss_breach"]
            or value["timed_out"]
            or duration > value["wall_time_limit_seconds"]
        ):
            raise ContractError(f"{label}: PASS command breached a resource limit")
        if state == "PASS" and value["network_isolated"] is not True:
            raise ContractError(f"{label}: test command was not network-isolated")

    validate_counts(counts, "result")
    if payload["state"] == "PASS" and counts["collected"] == 0:
        raise ContractError("PASS result has zero test collection")
    if payload["state"] == "PASS" and counts["selected"] == 0:
        raise ContractError("PASS result has zero tests selected")
    if len(steps) != len(cases):
        raise ContractError("case/step collection count mismatch")
    step_ids = [step["step_id"] for step in steps]
    case_ids = [case["case_id"] for case in cases]
    if len(set(step_ids)) != len(step_ids) or len(set(case_ids)) != len(case_ids):
        raise ContractError("duplicate case or step id")

    aggregate_counts = {key: 0 for key in COUNT_KEYS}
    aggregate_output = 0
    aggregate_captured_output = 0
    aggregate_command_rss = 0
    for case, step in zip(cases, steps):
        expected_case = {
            "case_id": step["step_id"],
            **{key: value for key, value in step.items() if key != "step_id"},
        }
        if case != expected_case:
            raise ContractError(
                "case entry must mirror its step with case_id (not step_id)"
            )
        validate_counts(step["counts"], f"step {step['step_id']}")
        validate_step_resource(
            step["resource"],
            f"step {step['step_id']}",
            step["duration_seconds"],
            step["state"],
        )
        if (
            step["selection_required"]
            and step["counts"]["selected"] == 0
            and step["state"] == "PASS"
        ):
            raise ContractError(f"step {step['step_id']}: zero tests selected")
        if step["state"] == "PASS" and (
            step["exit_code"] != 0
            or step["counts"]["failed"] != 0
            or step["counts"]["skipped"] != 0
            or step["counts"]["passed"] != step["counts"]["selected"]
        ):
            raise ContractError(
                f"step {step['step_id']}: PASS is inconsistent with its exit/counts"
            )
        if parse_time(step["started_at"]) > parse_time(step["finished_at"]):
            raise ContractError(f"step {step['step_id']}: timestamps are reversed")
        for key in COUNT_KEYS:
            aggregate_counts[key] += step["counts"][key]
        aggregate_output += step["resource"]["output_bytes"]
        aggregate_captured_output += step["resource"]["captured_output_bytes"]
        aggregate_command_rss = max(
            aggregate_command_rss, step["resource"]["max_rss_bytes"]
        )

    if counts != aggregate_counts:
        raise ContractError("result counts do not equal command counts")
    if resource_summary["commands_executed"] != len(steps):
        raise ContractError("executed command count does not match step records")
    expected_complete = (
        resource_summary["commands_executed"]
        == resource_summary["commands_expected"]
    )
    if resource_summary["commands_complete"] != expected_complete:
        raise ContractError("inconsistent command-completion flag")
    if resource_summary["output_bytes"] != aggregate_output:
        raise ContractError("row output total does not match command output")
    if resource_summary["captured_output_bytes"] != aggregate_captured_output:
        raise ContractError("row captured output total does not match command output")
    if resource_summary["captured_output_bytes"] > resource_summary["output_bytes"]:
        raise ContractError("row captured output exceeds observed output")
    expected_output_breach = (
        aggregate_output > resource_summary["row_output_limit_bytes"]
        or any(step["resource"]["output_breach"] for step in steps)
        or (
            not resource_summary["commands_complete"]
            and aggregate_output >= resource_summary["row_output_limit_bytes"]
        )
    )
    if resource_summary["output_breach"] != expected_output_breach:
        raise ContractError("inconsistent row output breach flag")
    expected_fixture_breach = (
        resource_summary["fixture_size_bytes"]
        > resource_summary["fixture_size_limit_bytes"]
    )
    if resource_summary["fixture_size_breach"] != expected_fixture_breach:
        raise ContractError("inconsistent fixture-size breach flag")
    expected_row_rss = max(
        aggregate_command_rss, resource_summary["runner_max_rss_bytes"]
    )
    if resource_summary["max_rss_bytes"] != expected_row_rss:
        raise ContractError("row max RSS does not cover runner and command RSS")
    expected_rss_breach = (
        resource_summary["max_rss_bytes"]
        > resource_summary["max_rss_limit_bytes"]
        or any(step["resource"]["rss_breach"] for step in steps)
    )
    if resource_summary["rss_breach"] != expected_rss_breach:
        raise ContractError("inconsistent row max RSS breach flag")
    expected_wall_breach = (
        payload["duration_seconds"] > resource_summary["wall_time_limit_seconds"]
        or (
            not resource_summary["commands_complete"]
            and payload["duration_seconds"]
            >= resource_summary["wall_time_limit_seconds"]
        )
    )
    if resource_summary["wall_time_breach"] != expected_wall_breach:
        raise ContractError("inconsistent row wall-time breach flag")
    step_strategies = {
        step["resource"]["network_guard_strategy"] for step in steps
    }
    if set(resource_summary["network_guard_strategies"]) != (
        step_strategies or {"unavailable"}
    ):
        raise ContractError(
            "row network guard strategies do not match command records"
        )
    if any(
        step["resource"]["address_space_limit_bytes"]
        != resource_summary["address_space_limit_bytes"]
        for step in steps
    ):
        raise ContractError(
            "row address-space limit does not match command records"
        )
    expected_network_isolated = bool(steps) and all(
        step["resource"]["network_isolated"] for step in steps
    )
    if resource_summary["network_isolated"] != expected_network_isolated:
        raise ContractError("inconsistent row network-isolation flag")

    if payload["evidence_mode"] == "required-ci":
        if payload["state"] == "PASS" and not payload["worktree_clean"]:
            raise ContractError("required CI evidence must use a clean worktree")
        if len(
            {
                payload["reviewed_sha"],
                payload["tested_sha"],
                payload["workflow_sha"],
            }
        ) != 1:
            raise ContractError(
                "required CI reviewed/tested/workflow SHA values must be identical"
            )
    elif not any(
        "not immutable evidence" in warning.lower()
        for warning in payload["diagnostic"]["warnings"]
    ):
        raise ContractError(
            "local-development result must say it is not immutable evidence"
        )
    if payload["required"] and payload["state"] in ("SKIP", "QUARANTINED"):
        raise ContractError("required result cannot be SKIP or QUARANTINED")
    if payload["required"] and counts["skipped"] and payload["state"] == "PASS":
        raise ContractError("required result contains skipped tests")
    if payload["state"] == "PASS":
        if counts["failed"] != 0 or counts["skipped"] != 0:
            raise ContractError("PASS result contains failed or skipped outcomes")
        if counts["passed"] != counts["selected"]:
            raise ContractError("PASS result did not pass every selected test")
        if any(case["state"] != "PASS" for case in cases):
            raise ContractError("PASS result contains non-PASS case")
        if not resource_summary["commands_complete"]:
            raise ContractError("PASS result did not execute every registered command")
        if (
            resource_summary["fixture_size_breach"]
            or resource_summary["output_breach"]
            or resource_summary["rss_breach"]
            or resource_summary["wall_time_breach"]
        ):
            raise ContractError("PASS result breached a row resource limit")
        if resource_summary["network_isolated"] is not True:
            raise ContractError("PASS row was not network-isolated")
        if payload["diagnostic"]["network_guard_self_test"] is not True:
            raise ContractError(
                "PASS result did not complete the network guard self-test"
            )
    if payload["state"] == "FAIL" and not any(
        case["state"] == "FAIL" for case in cases
    ):
        row_failure = (
            not resource_summary["commands_complete"]
            or resource_summary["fixture_size_breach"]
            or resource_summary["output_breach"]
            or resource_summary["rss_breach"]
            or resource_summary["wall_time_breach"]
            or (
                payload["evidence_mode"] == "required-ci"
                and not payload["worktree_clean"]
            )
        )
        if not row_failure:
            raise ContractError("FAIL result has no failed case or row breach")
    if payload["state"] == "INFRA_ERROR" and not any(
        case["state"] == "INFRA_ERROR" for case in cases
    ):
        if resource_summary["commands_complete"]:
            raise ContractError("INFRA_ERROR result has no infrastructure case")
    if payload["state"] != "PASS" and not payload["diagnostic"]["errors"]:
        raise ContractError("non-PASS result has no diagnostic error")
    for begin, end in (
        ("created_at", "finished_at"),
        ("started_at", "finished_at"),
    ):
        if parse_time(payload[begin]) > parse_time(payload[end]):
            raise ContractError(f"{begin} is after {end}")


def result_report_bytes(payload: dict[str, Any]) -> bytes:
    return canonical_bytes(payload)


def parse_sidecar(path: Path) -> str:
    try:
        first = path.read_text(encoding="utf-8").strip().split()
    except OSError as exc:
        raise ContractError(f"cannot read sidecar {path}: {exc}") from exc
    expected_name = path.name.removesuffix(".sha256")
    if (
        len(first) != 2
        or len(first[0]) != 64
        or any(c not in "0123456789abcdef" for c in first[0])
        or first[1] != expected_name
    ):
        raise ContractError(f"invalid sha256 sidecar {path}")
    return first[0]


def ensure_local_command(argv: list[str], repo: Path) -> list[str]:
    expanded = [str(sys.executable) if arg == "{python}" else arg for arg in argv]
    for arg in expanded:
        if "://" in arg or arg.startswith(("http:", "https:", "ssh:")):
            raise ContractError(f"network command is prohibited: {arg}")
    if not expanded:
        raise ContractError("empty command")
    if expanded[0] == sys.executable:
        if len(expanded) >= 2 and expanded[1] == "-m":
            if len(expanded) < 3 or expanded[2] not in {"pytest", "unittest"}:
                raise ContractError(f"Python module is not allowlisted: {expanded[2:]}")
        elif len(expanded) >= 2:
            candidate = Path(expanded[1])
            if candidate.is_absolute() or not (repo / candidate).resolve().is_relative_to(repo.resolve()):
                raise ContractError(f"command script escapes repository: {candidate}")
    elif Path(expanded[0]).name not in SAFE_COMMANDS:
        raise ContractError(f"command executable is not allowlisted: {expanded[0]}")
    if Path(expanded[0]).name == "cargo":
        validate_cargo_command(expanded, label="required host command")
    return expanded


def registered_row_commands(
    suites: dict[str, Any], row: dict[str, Any], repo: Path = ROOT
) -> list[tuple[str, list[str]]]:
    """Expand one row's immutable command list and reject duplicate IDs."""

    suite_by_id = {suite["suite_id"]: suite for suite in suites["suites"]}
    commands: list[tuple[str, list[str]]] = []
    seen: set[str] = set()
    for suite_id in row["suite_ids"]:
        suite = suite_by_id[suite_id]
        for command in suite["commands"]:
            command_id = f"{suite_id}.{command['command_id']}"
            if command_id in seen:
                raise ContractError(f"duplicate command id: {command_id}")
            seen.add(command_id)
            commands.append(
                (command_id, ensure_local_command(command["argv"], repo))
            )
    if not commands:
        raise ContractError(f"row {row['row_id']} collected zero commands")
    return commands


def isolated_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "CI": "true",
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "NO_NETWORK": "1",
        "ULLM_CI_NETWORK_DISABLED": "1",
        "ULLM_CI_MODEL_DISABLED": "1",
        "ULLM_CI_NO_MODEL": "1",
        "ULLM_CI_NO_GPU_FALLBACK": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "HIP_VISIBLE_DEVICES": "",
        "ROCR_VISIBLE_DEVICES": "",
        "JAX_PLATFORMS": "cpu",
        "CARGO_NET_OFFLINE": "true",
        "RUSTUP_AUTO_INSTALL": "0",
    })
    return env


def command_content_hash(steps: Iterable[dict[str, Any]]) -> str:
    descriptors = [{key: step[key] for key in ("step_id", "state", "exit_code", "stdout_sha256", "stderr_sha256")} for step in steps]
    return sha256_json(descriptors)


def command_hash(commands: Iterable[list[str]]) -> str:
    return sha256_json(list(commands))
